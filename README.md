
# RDO (Ready Designer One)

A node-based real-time video and graphics prototyping tool.  
Provides a TouchDesigner-style **TOP / CHOP node system** with a visual node editor.

> 🎯 **Current Main Feature: Blob Tracking**  
> Detects and tracks blobs in real time from image or video inputs.

---

> **⚠️ Project Status**  
> This project is in a very early stage.  
> This is the first Metal-based application developed in this project, so parts of the codebase may lack structure or refinement.  
> Developed with assistance from CODEX. Naming conventions (variables, functions, files) are not fully unified yet.  
> Unexpected bugs or crashes may occur. Refactoring and naming standardization are planned.

---

## 🎬 Demo

### 🧠 Workspace Demo (Main)

<p align="center">
  <a href="docs/media/RDO_workspace_video.mp4">
    <img src="docs/media/flower_workspace.png" width="900" alt="RDO Workspace Preview"/>
  </a>
</p>

<p align="center">
  <b>RDO Workspace Demo</b><br/>
  Node graph configuration · Real-time Blob Tracking · TOP/CHOP workflow
</p>

<p align="center">
  ▶️ <a href="docs/media/RDO_workspace_video.mp4">Watch Video</a>
</p>

---

## 🖥 Workspace Screenshot

<p align="center">
  <img src="docs/media/flower_workspace.png" width="900" alt="RDO Workspace Screenshot"/>
</p>

---

## 🎥 Output Examples

<div align="center">

<table>
<tr>
<td align="center">
<a href="docs/media/firework.mp4">
  <img src="docs/media/flower_workspace.png" width="420" alt="Firework Output"/>
</a><br/>
<b>Firework Output</b><br/>
<a href="docs/media/firework.mp4">Watch Video</a>
</td>

<td align="center">
<a href="docs/media/flower.mp4">
  <img src="docs/media/flower_workspace.png" width="420" alt="Flower Output"/>
</a><br/>
<b>Flower Output</b><br/>
<a href="docs/media/flower.mp4">Watch Video</a>
</td>
</tr>
</table>

</div>

---

## Key Features

- **Blob Tracking** – Main feature. Detects and tracks blobs from video/image input
- **Node Editor** – Visual graph editing based on ImNodes
- **TOP (Texture Operator)** – Image/video processing pipeline (blur, color correction, compositing, tracking, etc.)
- **CHOP (Channel Operator)** – Channel/signal processing (math, filters, trails, etc.)
- **Metal GPU** – High-performance rendering using macOS Metal
- **Video Engine** – Media playback and encoding support

---

## Tech Stack

| Category | Technology |
|----------|------------|
| Platform | macOS |
| GPU | Metal |
| UI | Dear ImGui, ImNodes |
| Language | C++, Objective-C++ |

---

## Project Structure

### Root Structure

```mermaid
flowchart TB
    RDO["📁 RDO (Root)"]

    RDO --> app["📁 RDO/"]
    RDO --> imgui["📁 imgui/"]
    RDO --> third["📁 third_party/"]
    RDO --> xcode["RDO.xcodeproj"]

    app --> core["📁 core/"]
    app --> nodes["📁 nodes/"]
    app --> f1["main.mm"]
    app --> f2["node_editor.mm · .h"]
    app --> f3["timeline.mm · .h"]
    app --> f4["texture_pool.mm · .h"]
    app --> f5["graph.h · gpu_batch_processor · performance_monitor · preview_settings"]

    core --> node_sys["📁 node_system/"]
    core --> gpu_dir["📁 gpu/"]
    core --> media_dir["📁 media/video/"]

    nodes --> TOP["📁 TOP/"]
    nodes --> CHOP["📁 CHOP/"]
    nodes --> SOP["📁 SOP/"]
    nodes --> DAT["📁 DAT/"]
    nodes --> COMP["📁 COMP/"]
    nodes --> AI["📁 AI/"]
    nodes --> eval["node_evaluator · output_node · node_types"]
```

### core/ Details

```mermaid
flowchart LR
    subgraph core["📁 core/"]
        subgraph node_system["node_system/"]
            nm["node_manager"]
            neo["node_execution_order"]
            rc["render_context"]
            port["port"]
            nb["node_base"]
            nr["node_registry"]
        end
        subgraph gpu["gpu/"]
            tvu["texture_view_utils"]
        end
        subgraph media["media/video/"]
            ve["video_engine"]
            vfe["video_file_encoder"]
        end
    end
```

### nodes/ Details

```mermaid
flowchart TB
    nodes["📁 nodes/"]

    nodes --> TOP["📁 TOP/"]
    nodes --> CHOP["📁 CHOP/"]
    nodes --> SOP["SOP/"]
    nodes --> DAT["DAT/"]
    nodes --> COMP["COMP/"]
    nodes --> AI["AI/"]
    nodes --> shared["node_evaluator · output_node · node_types"]

    subgraph CHOP_detail["CHOP/"]
        CHOP_analysis["analysis/ (blob_track_info)"]
        CHOP_filter["filter/ (trail)"]
        CHOP_gen["generator/ (time)"]
        CHOP_math["math/ (add, multiply, sine)"]
    end

    CHOP --> CHOP_detail
```

---

## Build Instructions

1. Open `RDO.xcodeproj` in **Xcode**
2. Select the **RDO** scheme
3. Press **Run (⌘R)**

### Requirements
- macOS
- Latest Xcode
- Metal-supported Mac

---

## License

Check the license applied to each project component.  
Third-party libraries (Dear ImGui, ImNodes, etc.) follow their respective licenses.
