// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { mount } from "svelte";
import App from "./App.svelte";
import "./app.css";
import { installHints } from "./lib/hint";
import { applyTheme, readTheme } from "./lib/theme";

applyTheme(readTheme());
installHints();

mount(App, { target: document.getElementById("app")! });
