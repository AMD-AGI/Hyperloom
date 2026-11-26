// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { Edge, Node } from "@xyflow/svelte";
import type { GraphNode } from "../types";
import type { PairEdge } from "./edges";
import type { LabelSpot } from "./layout";

export type AgentNodeType = Node<{ node: GraphNode; selected: boolean; faded: boolean }, "agent">;
export type BusEdgeType = Edge<
  { pair: PairEdge; recent: number; faded: boolean; path: string; label: LabelSpot | null },
  "bus"
>;
