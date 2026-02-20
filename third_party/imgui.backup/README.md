# ImGui SDK

이 디렉토리는 Dear ImGui 라이브러리를 포함합니다.

## 구조

- `imgui.h`, `imgui.cpp` 등: ImGui 코어 파일
- `backends/`: 플랫폼별 백엔드 구현
  - `imgui_impl_metal.h/mm`: Metal 렌더링 백엔드
  - `imgui_impl_osx.h/mm`: macOS 플랫폼 백엔드
- `imconfig.h`: ImGui 설정 파일

## 사용법

프로젝트에서 다음과 같이 include:

```cpp
#include "imconfig.h"
#include "imgui.h"
#include "imgui_impl_metal.h"
#include "imgui_impl_osx.h"
```

