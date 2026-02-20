#!/bin/bash

# Metal Validation 없이 실행 (크래시 우회)

# Metal validation 비활성화
export METAL_DEVICE_WRAPPER_TYPE=0
export MTL_DEBUG_LAYER=0

BUILD_DIR="/Users/junu/Library/Developer/Xcode/DerivedData/RDO-cwcvxrbrmkkkrkckagldbbpsiadk/Build/Products/Debug"

echo "========================================="
echo "Running RDO without Metal Validation"
echo "Warning: This hides validation errors"
echo "========================================="
echo ""

if [ -f "$BUILD_DIR/RDO" ]; then
    "$BUILD_DIR/RDO"
else
    echo "Error: Build not found"
    echo "Run: xcodebuild -project RDO.xcodeproj -scheme RDO -configuration Debug build"
    exit 1
fi
