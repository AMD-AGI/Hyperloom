// The program's own annotations as ranges (RUPPELL_NVTX=1; off by default; roadmap-v2 M7),
// shared by the platform tools: NVTX on CUDA (tool_cuda.cpp, CUPTI's NVTX callbacks) and ROCTX
// on ROCm (tool.cpp, the profiler SDK's marker callbacks). Header-only, included once by each.
//
// A range is written when it closes (a push at its pop, a start at its end) and a mark when it
// happens, timestamped by the tool, so ranges nest exactly with its API and graph records. As
// in NVTX, push ranges are per thread and per domain: a domain's pop closes that domain's
// innermost push on the thread. Records go to ranges.bin (ruppell_range_rec), names to
// ranges.tsv (id, domain, text). RUPPELL_NVTX_DOMAINS (comma-separated) keeps only the domains
// named; the unnamed domain is "default", and ROCTX's is "roctx".
#pragma once

#include "records.h"

#include <stdio_ext.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ranges
{
inline bool                  enabled  = false;
inline FILE*                 f_ranges = nullptr;
inline FILE*                 f_names  = nullptr;
inline std::mutex&           m        = *new std::mutex();  // the files and every table below
inline std::atomic<uint64_t> n_ranges{0}, n_open{0}, n_unmatched{0};

// leaked, as the tools' globals are: the exit handler may run after static destructors
inline auto& names    = *new std::unordered_map<std::string, uint64_t>();  // "domain\ttext" -> id
inline auto& domains  = *new std::unordered_map<uint64_t, std::string>();  // handle -> name
inline auto& strings  = *new std::unordered_map<uint64_t, std::string>();  // registered handle -> text
inline auto& keep     = *new std::unordered_set<std::string>();            // empty: every domain
inline auto& started  = *new std::unordered_map<uint64_t, ruppell_range_rec>();  // start ranges by id

struct attrs
{
    uint64_t payload      = 0;
    uint32_t payload_type = 0;
    uint32_t category     = 0;
};

struct open_range
{
    ruppell_range_rec r;
    uint64_t          domain;
};
// push ranges open on this thread, innermost last; a range of a domain not kept has name 0,
// so its pop still finds it
inline thread_local std::vector<open_range> tl_open;

inline bool
env_on()
{
    const char* v = std::getenv("RUPPELL_NVTX");
    return v && *v && std::string(v) != "0";
}

inline void
open(const std::string& dir)
{
    if(!env_on()) return;
    if(const char* d = std::getenv("RUPPELL_NVTX_DOMAINS"))
    {
        std::string s(d), cur;
        for(char c : s + ",")
            if(c == ',')
            {
                if(!cur.empty()) keep.insert(cur);
                cur.clear();
            }
            else if(c != ' ')
                cur += c;
    }
    f_ranges = std::fopen((dir + "/ranges.bin").c_str(), "wb");
    f_names  = std::fopen((dir + "/ranges.tsv").c_str(), "w");
    enabled  = f_ranges && f_names;
    // records are written on the annotating thread: a large buffer keeps its writes rare (on a
    // synchronously mounted NFS a buffer-sized write blocked it 0.75-1 ms; MI350, 2026-10-20)
    for(FILE* f : {f_ranges, f_names})
        if(f) std::setvbuf(f, nullptr, _IOFBF, 1u << 20);
}

// Drop a stream inherited across fork() without writing it (as cpu_tree::drop_inherited)
inline void
drop_inherited(FILE*& f)
{
    if(!f) return;
    __fpurge(f);
    std::fclose(f);
    f = nullptr;
}

inline void
forget_in_child()
{
    enabled = false;
    drop_inherited(f_ranges);
    drop_inherited(f_names);
}

// tabs and newlines would break ranges.tsv
inline std::string
clean(std::string s)
{
    for(char& c : s)
        if(c == '\t' || c == '\n' || c == '\r') c = ' ';
    return s;
}

// a wide string as UTF-8 (NVTX's W calls; rare)
inline std::string
from_wide(const wchar_t* w)
{
    std::string s;
    for(; w && *w; ++w)
    {
        uint32_t c = static_cast<uint32_t>(*w);
        if(c < 0x80)
            s += static_cast<char>(c);
        else if(c < 0x800)
        {
            s += static_cast<char>(0xC0 | (c >> 6));
            s += static_cast<char>(0x80 | (c & 0x3F));
        }
        else if(c < 0x10000)
        {
            s += static_cast<char>(0xE0 | (c >> 12));
            s += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
            s += static_cast<char>(0x80 | (c & 0x3F));
        }
        else
        {
            s += static_cast<char>(0xF0 | (c >> 18));
            s += static_cast<char>(0x80 | ((c >> 12) & 0x3F));
            s += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
            s += static_cast<char>(0x80 | (c & 0x3F));
        }
    }
    return s;
}

inline void
domain_created(uint64_t handle, const std::string& name)
{
    std::lock_guard<std::mutex> lk(m);
    domains[handle] = clean(name);
}

inline void
string_registered(uint64_t handle, const std::string& text)
{
    std::lock_guard<std::mutex> lk(m);
    strings[handle] = text;
}

inline std::string
registered(uint64_t handle)
{
    std::lock_guard<std::mutex> lk(m);
    auto it = strings.find(handle);
    return it == strings.end() ? std::string() : it->second;
}

// the name id of (domain, text), written to ranges.tsv the first time; 0 when the domain is
// not kept. Domain 0 is the unnamed one.
inline uint64_t
intern(uint64_t domain, const std::string& text, const char* fixed_domain = nullptr)
{
    std::lock_guard<std::mutex> lk(m);
    std::string dn;
    if(fixed_domain)
        dn = fixed_domain;
    else if(domain == 0)
        dn = "default";
    else
    {
        auto it = domains.find(domain);
        dn      = it != domains.end() ? it->second : "domain@" + std::to_string((unsigned long long) domain);
    }
    if(!keep.empty() && !keep.count(dn)) return 0;
    std::string key = dn + "\t" + clean(text);
    auto        it  = names.find(key);
    if(it != names.end()) return it->second;
    uint64_t id = names.size() + 1;
    names.emplace(key, id);
    if(f_names) std::fprintf(f_names, "%llu\t%s\n", (unsigned long long) id, key.c_str());
    return id;
}

inline void
write(const ruppell_range_rec& r)
{
    std::lock_guard<std::mutex> lk(m);
    if(f_ranges) std::fwrite(&r, sizeof(r), 1, f_ranges);
    ++n_ranges;
}

inline void
push(uint64_t domain, uint64_t name, const attrs& a, uint64_t now, uint64_t tid)
{
    ruppell_range_rec r{now, 0, tid, 0, name, a.payload, RUPPELL_RANGE_PUSH, (uint32_t) tl_open.size(),
                        a.payload_type, a.category};
    tl_open.push_back({r, domain});
    ++n_open;
}

inline void
pop(uint64_t domain, uint64_t now, uint64_t tid)
{
    for(size_t i = tl_open.size(); i-- > 0;)
        if(tl_open[i].domain == domain)
        {
            ruppell_range_rec r = tl_open[i].r;
            tl_open.erase(tl_open.begin() + i);
            --n_open;
            if(!r.name) return;
            r.t1            = now;
            r.end_thread_id = tid;
            write(r);
            return;
        }
    ++n_unmatched;
}

inline void
start(uint64_t id, uint64_t name, const attrs& a, uint64_t now, uint64_t tid)
{
    std::lock_guard<std::mutex> lk(m);
    started[id] = {now, 0, tid, 0, name, a.payload, RUPPELL_RANGE_START, (uint32_t) tl_open.size(),
                   a.payload_type, a.category};
    ++n_open;
}

inline void
end(uint64_t id, uint64_t now, uint64_t tid)
{
    ruppell_range_rec r{};
    {
        std::lock_guard<std::mutex> lk(m);
        auto it = started.find(id);
        if(it == started.end())
        {
            ++n_unmatched;
            return;
        }
        r = it->second;
        started.erase(it);
        --n_open;
    }
    if(!r.name) return;
    r.t1            = now;
    r.end_thread_id = tid;
    write(r);
}

inline void
mark(uint64_t name, const attrs& a, uint64_t now, uint64_t tid)
{
    if(!name) return;
    write({now, now, tid, tid, name, a.payload, RUPPELL_RANGE_MARK, (uint32_t) tl_open.size(), a.payload_type,
           a.category});
}

// for meta.json: ranges written, ranges still open, and pops or ends with nothing open
inline std::string
meta_json(const char* how)
{
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "{\"enabled\": %s, \"source\": \"%s\", \"recorded\": %llu, \"open\": %llu, \"unmatched_ends\": %llu}",
                  enabled ? "true" : "false", how, (unsigned long long) n_ranges, (unsigned long long) n_open,
                  (unsigned long long) n_unmatched);
    return buf;
}

inline void
flush()
{
    std::lock_guard<std::mutex> lk(m);
    for(FILE* f : {f_ranges, f_names})
        if(f) std::fflush(f);
}

inline void
close()
{
    std::lock_guard<std::mutex> lk(m);
    for(FILE** f : {&f_ranges, &f_names})
        if(*f)
        {
            std::fclose(*f);
            *f = nullptr;
        }
    enabled = false;
}
}  // namespace ranges
