# Audit — issue #28 : `--kv-cache-dtype fp8` inutilisable sur Qwen2.5

Journal unique de l'audit/investigation (branche `issue28-kv-fp8-offset`,
livrée via PR #30 ; options 1 puis 1+2).
Constat de départ, mesures, design, implémentation, résultats e2e, limites.

---

## 0. Option 2 — scale per-canal du résidu (ajout postérieur)

Complément de l'option 1 (même schéma que « mean removed + per-(head,channel)
scale » du tableau section 3) : chaque canal du résidu est normalisé à sa
propre pleine gamme fp8 au store, et re-multiplié à la lecture. C'est la
première option qui touche les kernels — volontairement minimale :

- **Store** (`triton_reshape_and_cache_flash.py`) : `tile_pos` énumère déjà
  (head, dim) dans l'ordre naturel `[Hk, D]` → la charge per-canal est un
  flat load à `pc_ptr + tile_pos` avant la division fp8 ; kwarg optionnel
  `k_scale_channel` + constexpr `HAS_K_SCALE_CHANNEL`. Le variant `_diffkv`
  est du code mort (zéro appelant) — non touché.
- **Lecture** (`triton_unified_attention.py`) : K est chargé en tuiles
  `[HEAD_SIZE, TILE_SIZE]` dont l'axe ligne est le canal → charge
  loop-invariante `pc[kv_head_idx*HEAD_SIZE + offs_d]` avant la boucle de
  tuiles, multipliée au site du descale scalaire (branches fp8-K / Q non-fp8,
  kernels 2D et 3D) ; kwarg `k_descale_channel` + constexpr `USE_PC_K_SCALE`.
  Un seul appel `unified_attention` couvre prefill+decode.
- **Backend** (`triton_attn.py`) : les deux sites passent
  `getattr(layer, "_kv_k_pc_scale", None)`.
- **Plugin** (`kv_offsets.py`) : clé `"pc_scales"` du fichier (optionnelle —
  absente = option 1 seule, compat ascendante) ; quand présente, `_k_scale`
  reste à 1.0 (le scale per-canal **remplace** le per-tensor, sémantique de
  l'issue). Calibration : `pc_scales` = `res_max/448` par (head, canal) — la
  carte était déjà calculée en passe 2 du probe, zéro coût GPU en plus.
- Ces changements de l'arbre `vllm/` sont capturés dans
  `patches/vllm/triton-attn-pc-scale.patch` (appliqué par
  `tools/patch_vllm.py`, ordre : après `native-cache-ops`).

### Vérifications option 2

- **Check kernel direct** (`run/kv_pc_kernel_check.py`) : store round-trip
  (layouts flat 4D ET head-major 5D) et lecture 2D, on/off, **bit-exacts**
  vs référence torch.
- **Offline** (couche 27 @ 8192, sémantique runtime) :
  baseline 0.890/0.240 → option 1 : 0.049-0.058/0.033 → **option 1+2 :
  0.047/0.017** (le rms-rel 0.017 est exactement la valeur du tableau de
  l'issue ; le max-rel partait déjà bas grâce au mu calibré).
- **E2E** (`run/kv_bench.py`, même réf fp16 que la section 7) :

| ctx | option 1 (agree / decode tok/s) | option 1+2 (agree / decode tok/s) |
| --- | --- | --- |
| 125 | 32 % / 82.2 | 31 % / 81.8 |
| 2045 | 43 % / 78.6 | 87 % / 78.6 |
| 8189 | 2-20 % / 71.5 | 9 % / 71.5 |

  ⚠️ **`agree_frac` est une métrique bruitée** : trois runs identiques
  (même binaire, même config, fichier option 1) ont donné 31/4/20 puis
  31/4/3 — les ctx 125 et 2045 se reproduisent, le 8189 non. Cause :
  nondéterminisme run-à-run du moteur lui-même (très probablement les
  réductions atomiques du GEMM GPTQ int4), que le décodage glouton
  amplifie en cascades d'argmax sur quasi-égalités. Les tirages ci-dessus
  sont donc indicatifs ; la comparaison fiable est l'offline (rms-rel
  0.033 → 0.017, déterministe) et la qualité de texte, stable et fluide
  dans **tous** les runs de toutes les configs depuis l'option 1.

  Perf decode identique (les charges per-canal sont loop-invariantes côté
  lecture, un flat load côté store). Compat : l'ancien fichier sans
  `pc_scales` garde le comportement option 1 (« 0 with per-channel scale »
  en log, textes cohérents).

---

## 1. Symptôme

`--kv-cache-dtype fp8` sur Qwen2.5-7B-Instruct-GPTQ-Int4 (7900 XT / gfx1100,
backend TRITON_ATTN) dégrade immédiatement la génération : `run/kv_bench.py`
signale `agree_frac` ≈ 0 vs la référence fp16 dès les premiers tokens
(`first_divergence` ≈ 0) à toutes les longueurs de contexte. Le même modèle en
fp16 (`auto`) est correct.

## 2. Mesures (issues de #25/#29, reproduites via les outils du repo)

Store fp8 et lectures Triton **bit-exacts** (2D et 3D identiques) → l'erreur
vient de la quantization elle-même. V est fine en per-tensor ; c'est **K**.

- Amplitudes (`run/kv_scale_probe.py`, signature corrigée en #29) :
  max|K| 419-423, max|V| 72.5, max|Q| 80, identiques à 128/2k/8k tokens.
- Par (kv head, canal), max|K| toutes couches confondues : médiane 6.9,
  p95 93, 22/512 paires > 100 — toutes dans les canaux 58-63 et 123-127 des
  kv heads 0 et 1, c.-à-d. les **derniers canaux de chaque moitié RoPE** (les
  fréquences rotatoires les plus basses, qui tournent à peine sur 8k
  positions). Par couche : 27 (423), 0 (173), le reste < 80.
- **Nature du problème : offset, pas échelle.** Dans ces canaux K ≈ biais
  k_proj quasi constant + petit résidu. fp8 = précision relative : ~420
  stocké avec erreur absolue jusqu'à 16 **quelle que soit l'échelle** →
  bruit de logits dépendant de la position. Rien qui ne re-scale ne peut
  aider ; une rotation (Hadamard) ne fait qu'étaler l'offset.

## 3. Comparaison offline des schémas (`run/kv_quant_schemes.py` sur dump réel,
`VLLM_KV_PROBE_DUMP` + `VLLM_KV_PROBE_DUMP_LAYER`)

max-rel / rms-rel de l'output d'attention (dernières 256 requêtes, 8192
tokens) vs référence fp32 :

| Schéma K (V exact sauf mention) | couche 12 (douce, max K 21) | couche 27 (max K 423) | couche 0 (max K 173) |
| --- | --- | --- | --- |
| fp8 e4m3 per-tensor 1.0 (= vLLM today) | 0.149 / 0.094 | 0.911 / 0.237 | 1.036 / 0.627 |
| fp8 per-tensor pleine échelle | 0.063 / 0.058 | 0.709 / 0.362 | 1.190 / 0.693 |
| fp8 per-(head,canal) | 0.056 / 0.063 | 0.937 / 0.426 | 0.979 / 0.575 |
| fp8 per-token | 0.051 / 0.041 | 0.879 / 0.444 | 1.171 / 0.708 |
| fp8 e5m2 per-tensor 1.0 | 0.156 / 0.158 | 0.974 / 0.540 | 1.029 / 0.876 |
| fp8 après rotation Hadamard | 0.032 / 0.019 | 0.818 / 0.446 | 1.024 / 0.599 |
| **fp8 moyenne retirée + per-tensor** | 0.115 / 0.092 | **0.070 / 0.045** | **0.079 / 0.026** |
| fp8 moyenne retirée + scale per-(head,canal) du résidu | 0.039 / 0.017 | 0.031 / 0.017 | 0.061 / 0.027 |
| int8 asym per-(head,canal) | 0.007 / 0.004 | 0.008 / 0.003 | 0.011 / 0.004 |

Conclusion : **retirer la moyenne par canal** est le seul schéma fp8 qui
corrige les couches aberrantes (0.9-1.0 → 0.07). Le scale per-canal du
résidu (option 2) gagne encore un facteur 2 mais exige de toucher les
kernels de lecture ET d'écriture — hors périmètre (validé avec l'utilisateur).

**Pourquoi ne jamais rajouter la moyenne** : q·mean est identique pour toutes
les clés d'une même requête → annule dans le softmax. Soustraire sans
remettre est donc mathématiquement neutre, et les mesures confirment
(schéma « mean removed, not added back » : nombres identiques).

## 4. Design retenu (option 1 de l'issue)

- **Calibration** : moyenne de K par (couche, kv head, canal) sur un prompt
  de calibration → fichier `.pt` {model, offsets {layer: [Hk, D]},
  k_scales {layer: max|résidu|/448}}.
- **Runtime, store uniquement** : `key = key - offset[layer]` avant le store
  fp8 ; `_k_scale[layer] = max|résidu|/448` (le scale per-tensor par couche
  existant). Lecture inchangée (descale × k_scale rend le résidu ; softmax
  annule q·mean). Zéro kernel modifié.
- Env vars : calibration `VLLM_KV_PROBE_MEAN=<out.pt>` (probe), runtime
  `VLLM_WIN_KV_OFFSETS=<file.pt>` (plugin).

### Points de code validés pendant l'audit

- Store : `TritonAttentionImpl.do_kv_cache_update`
  (`vllm/vllm/v1/attention/backends/triton_attn.py:581`) ; `key` y est
  post-RoPE, dtype natif, `[T, Hk, D]` ; les vues fp8 (:597-599) portent sur
  `kv_cache`, pas sur `key` → soustraire avant l'appel est correct. Le path
  natif HIP est déjà bypassé en fp8 (:603-607).
- Lecture : `triton_unified_attention.py` descale `K_load.to(fp32) *
  k_scale` (:266, :621) → renvoie le résidu. Aucun changement.
- Chemins **non atteignables** sous TRITON_ATTN, documentés non touchés :
  kernel `diffkv` (uniquement `static_sink_attention.py` /
  `flash_attn_diffkv.py`) et `do_rope_and_kv_cache_update` (requiert aiter,
  absent sur Windows).
- MLA : classe séparée, jamais vue par le wrapper `Attention`. Couches sans
  entrée dans le fichier → warn once, fp8 brut.
- Garde probe : le dummy run de profiling passe `attn_metadata=None`
  (`triton_attn.py:443`) → l'accumulation de moyenne le saute (un max le
  tolérait, une moyenne non).
- Style plugin : monkey-patchs installés depuis
  `platform.check_and_update_config`, précédents `kv_scales.py` et
  `run/s5_bench/perf_ablation.py` (qui patche déjà `do_kv_cache_update`).

## 5. Implémentation

- `run/kv_scale_probe.py` : mode `VLLM_KV_PROBE_MEAN` (double passe :
  sommes → moyenne, puis max du résidu ; écrit le fichier).
- `windows_rocm_plugin/vllm_windows_rocm/kv_offsets.py` (nouveau) : wrap
  `Attention.process_weights_after_loading` (offset stashé par couche +
  `_k_scale` par couche) et `TritonAttentionImpl.do_kv_cache_update`
  (soustraction, hors-place, garde fp8).
- `windows_rocm_plugin/vllm_windows_rocm/platform.py` : enregistrement
  `kv_offsets.register()` (+ doc : ne pas combiner avec `VLLM_WIN_KV_KSCALE`).
- Soustraction hors-place : `unified_kv_cache_update` déclare
  `mutates_args=[]` → muter `key` du caller violerait le contrat compile.

## 6. Blocage environnement rencontré pendant l'audit (résolu)

Au premier lancement du probe, `from vllm import LLM` échouait :
`xgrammar` 0.2.8 charge ses bindings via **apache-tvm-ffi 0.1.14.post1**, dont
le loader DSO Windows passe par l'API ANSI : **tout DLL situé sur un chemin
non-ASCII échoue à se charger** (`C:\Users\PC-Clément\...` → « Failed to load
dynamic shared library »). Vérifié : le même DLL se charge via ctypes (API
wide) et via tvm_ffi depuis un chemin ASCII — cause déterministe, pas de
version corrigée en amont (0.1.14.post1 = dernière).

Contournement (env, hors repo) : `.venv/Lib/site-packages/sitecustomize.py`
convertit le chemin en **forme courte 8.3** (ASCII pur) avant
`tvm_ffi.load_module`, avec extension remise en minuscules (le registre de
loaders tvm_ffi est sensible à la casse : `.DLL` ≠ `.dll`). Auto-chargé par
`site.py` à chaque démarrage de l'interpréteur du venv, no-op si tvm_ffi est
absent. (Les runs de la veille avaient fonctionné dans un état système
différent ; cause non reproduite mais le mécanisme ci-dessus est vérifiable
à volonté.)

Deuxième blocage, même session : la machine a **deux GPU** (iGPU APU
gfx10.3 « AMD Radeon(TM) Graphics » en device 0 + RX 7800 XT gfx1100 en
device 1). HIP a énuméré l'iGPU en 0 → `HIP error: device kernel image is
invalid` (pas de code object pour gfx10.3 dans le torch ROCm nightly). Fix :
`HIP_VISIBLE_DEVICES=1` sur chaque invocation.

## 7. Résultats e2e (7800 XT, Qwen2.5-7B-Instruct-GPTQ-Int4, `run/kv_bench.py`)

Calibration (`VLLM_KV_PROBE_MEAN`, ctxs 128/2048/8192) : amplitudes
conformes à l'issue (max|K| 419.5-422.8 ; couches 27 puis 0 dominantes).
Après retrait de la moyenne, le résidu max par couche tombe à ~24
(k_scale max 0.0545, couche 1) — la couche 27 passe de 423 à ~15.5.
Fichier : 28 couches × [4, 128] fp32 + 28 k_scales.

| ctx | dtype | prefill | decode tok/s | agree | 1re diverge (token) |
| --- | --- | --- | --- | --- | --- |
| 125 | fp16 (réf) | 0.088 s | 83.0 | — | — |
| 2045 | fp16 (réf) | 0.584 s | 81.2 | — | — |
| 8189 | fp16 (réf) | 3.154 s | 75.6 | — | — |
| 125 | fp8 brut | 0.089 s | 82.5 | 2 % | 0 |
| 2045 | fp8 brut | 0.662 s | 79.1 | 2 % | 0 |
| 8189 | fp8 brut | 4.276 s | 71.7 | 2 % | 0 |
| 125 | **fp8+offsets** | 0.089 s | 82.2 | 32 % | 40 |
| 2045 | **fp8+offsets** | 0.663 s | 78.6 | 43 % | 0 |
| 8189 | **fp8+offsets** | 4.317 s | 71.5 | 2 % | 2 |

**Qualité de sortie** (le critère qui compte — l'exactitude token-à-token
reste impossible pour tout schéma fp8 à ~5 % de bruit, cf. tableau de
l'issue) :

- fp8 brut : charabia immédiat à toutes les longueurs
  (« fixed fixed fixedindhoven pipelines fixed to programm maic shaders »).
- fp8+offsets : résumés **fluides et fidèles** aux trois longueurs, ex. à
  8189 : « GPU computing started with the transition from fixed function
  graphics pipelines to programmable shaders in the late 1990s, evolved
  through the GPGPU movement and the introduction of CUDA, and is now
  dominated by matrix… » — vs référence fp16 : même contenu.

**Perf** : decode au bruit près du fp8 brut (une soustraction + un scale
par couche au store) ; le surcoût prefill du fp8 (4.28 s vs 3.15 s à 8k)
préexiste (présent dans la baseline fp8 sans offsets : 4.276 s).

Wrapper vérifié installé dans les logs : « K offsets installed from …
(28 layers, model 'Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4') », zéro warning
« no K offset ».

Note de méthode : le critère initialement prévu (« agree ≈ plafond fp16 »)
était mal calibré — même le meilleur schéma fp8 de l'issue (0.03-0.07
max-rel) fait basculer l'argmax sur des quasi-égalités. Le critère tenu :
sortie cohérente + contenu aligné sur la réf fp16 + drift perf nul.

**Contre-vérification offline** (couche 27 @ 8192, dump réel, sémantique
runtime exacte — store `fp8((K−mu)/k_scale)`, lecture `résidu × k_scale`,
moyenne jamais rajoutée) :

| Config | max-rel / rms-rel |
| --- | --- |
| fp8 per-tensor 1.0 (= vLLM aujourd'hui) | 0.890 / 0.240 |
| **runtime : mu du fichier + k_scale 0.0347** | **0.058 / 0.033** |
| schéma : mu du dump + échelle pleine gamme 0.0334 | 0.076 / 0.052 |

Le mu calibré (moyenné sur 128/2k/8k sur les deux passes) est même
légèrement meilleur que le mu d'un dump mono-couche. Cohérent avec le
tableau de l'issue (0.911 → ~0.07).

## 8. Limites

- mu dépend du prompt de calibration (moyenne sur les positions d'un texte
  représentatif ; un prompt très différent décale les résidus).
- k_scale per-tensor sur le résidu **sans headroom** : un outlier neuf sature
  un canal. Upgrade path : multiplicateur headroom (une ligne).
- int8 asymétrique par canal resterait le format « juste » (< 0.011 partout)
  mais demande un nouveau cache dtype vLLM (option 3, non planifiée).

## 9. Reproduire

```bash
cd run
# GPU dGPU = device 1 sur cette machine (iGPU en 0)
export HIP_VISIBLE_DEVICES=1
# calibration
VLLM_KV_CTXS=128,2048,8192 VLLM_KV_PROBE_MEAN=kv_offsets_qwen25_7b_gptq.pt python kv_scale_probe.py
# référence fp16 puis fp8 brut puis fp8+offsets
VLLM_KV_DTYPE=auto VLLM_KV_SAVE_REF=1 python kv_bench.py
VLLM_KV_DTYPE=fp8 python kv_bench.py
VLLM_KV_DTYPE=fp8 VLLM_WIN_KV_OFFSETS=kv_offsets_qwen25_7b_gptq.pt python kv_bench.py
```
