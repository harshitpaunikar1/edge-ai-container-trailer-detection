# Edge AI Container & Trailer Detection Diagrams

Generated on 2026-04-26T04:29:37Z from README narrative plus project blueprint requirements.

## Edge detection pipeline

```mermaid
flowchart TD
    N1["Step 1\nClarified stakeholder goals and user stories; defined detection targets (container"]
    N2["Step 2\nBuilt representative dataset from wall-mounted cameras; curated scenes across day/"]
    N1 --> N2
    N3["Step 3\nTrained object-detection model and optimised for camera hardware; applied quantisa"]
    N2 --> N3
    N4["Step 4\nValidated with hold-out data and live pilots; monitored precision/recall, tuned th"]
    N3 --> N4
    N5["Step 5\nEngineered edge pipeline to extract events, cache locally, transmit derived data t"]
    N4 --> N5
```

## Camera-to-event flow diagram

```mermaid
flowchart LR
    N1["Inputs\nImages or camera frames entering the inference workflow"]
    N2["Decision Layer\nCamera-to-event flow diagram"]
    N1 --> N2
    N3["User Surface\nOperator-facing UI or dashboard surface described in the README"]
    N2 --> N3
    N4["Business Outcome\nInference or response latency"]
    N3 --> N4
```

## Evidence Gap Map

```mermaid
flowchart LR
    N1["Present\nREADME, diagrams.md, local SVG assets"]
    N2["Missing\nSource code, screenshots, raw datasets"]
    N1 --> N2
    N3["Next Task\nReplace inferred notes with checked-in artifacts"]
    N2 --> N3
```
