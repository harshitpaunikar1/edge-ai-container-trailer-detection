# Edge AI Container & Trailer Detection

> **Domain:** Logistics & Transportation

## Overview

A large yard operation required real-time visibility of containers and truck trailers without expanding network infrastructure or headcount. Manual checks were slow and inconsistent; camera feeds were underutilized with varying data quality across locations and lighting conditions. Business users lacked timely, consistent status data for planning gates, docks, and dispatches. Without automation, operations faced prolonged turnaround times, higher detention/demurrage costs, compliance gaps, and safety incidents. The objective involved detecting key attributes on-camera, processing at the edge, and delivering structured events to downstream systems for planning and auditing.

## Approach

- Clarified stakeholder goals and user stories; defined detection targets (container presence, trailer status, lane/zone context) and acceptance criteria
- Built representative dataset from wall-mounted cameras; curated scenes across day/night, weather, and occlusions; annotated images for supervised training
- Trained object-detection model and optimised for camera hardware; applied quantisation to meet on-device latency and power limits
- Validated with hold-out data and live pilots; monitored precision/recall, tuned thresholds, hardened against edge cases
- Engineered edge pipeline to extract events, cache locally, transmit derived data to client systems with retries and health checks
- Delivered in agile sprints with demos; documented results; managed tasks across internal teams and client stakeholders

## Skills & Technologies

- Computer Vision
- Edge AI Deployment
- Object Detection
- Model Quantization
- Data Annotation
- Model Evaluation
- Data Analysis
- Event-Driven Integration
- Stakeholder Presentation
- Agile Task Management
