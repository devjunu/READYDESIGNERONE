#!/bin/bash

# TOP Nodes
sed -i '' 's/REGISTER_NODE(MovieFileIn, "TOP\/IO"/REGISTER_NODE(MovieFileIn, "Movie File In", "TOP\/IO"/g' nodes/TOP/movie_file_in/movie_file_in_node.mm
sed -i '' 's/REGISTER_NODE(VideoFileLoader, "TOP\/IO"/REGISTER_NODE(MovieFileIn, "Movie File In", "TOP\/IO"/g' nodes/TOP/movie_file_in/movie_file_in_node.mm

sed -i '' 's/REGISTER_NODE(BackgroundSubtract, "TOP\/Filter"/REGISTER_NODE(BackgroundSubtract, "Background Subtract", "TOP\/Composite"/g' nodes/TOP/background_subtract/background_subtract_node.mm

sed -i '' 's/REGISTER_NODE(BlobTrack, "TOP\/Analysis"/REGISTER_NODE(BlobTrack, "Blob Track", "TOP\/Analysis"/g' nodes/TOP/blob_track/blob_track_node.mm

sed -i '' 's/REGISTER_NODE(Crop, "TOP\/Transform"/REGISTER_NODE(Crop, "Crop", "TOP\/Transform"/g' nodes/TOP/crop/crop_node.mm

sed -i '' 's/REGISTER_NODE(ColorCorrection, "TOP\/Color"/REGISTER_NODE(ColorCorrection, "Color Correction", "TOP\/Filter"/g' nodes/TOP/color_correction/color_correction_node.mm

sed -i '' 's/REGISTER_NODE(Over, "TOP\/Composite"/REGISTER_NODE(Over, "Over", "TOP\/Composite"/g' nodes/TOP/over/over_node.mm

sed -i '' 's/REGISTER_NODE(Flip, "TOP\/Transform"/REGISTER_NODE(Flip, "Flip", "TOP\/Transform"/g' nodes/TOP/flip/flip_node.mm

sed -i '' 's/REGISTER_NODE(Lookup, "TOP\/Color"/REGISTER_NODE(Lookup, "Lookup", "TOP\/Filter"/g' nodes/TOP/lookup/lookup_node.mm

sed -i '' 's/REGISTER_NODE(Edge, "TOP\/Filter"/REGISTER_NODE(Edge, "Edge", "TOP\/Filter"/g' nodes/TOP/edge/edge_node.mm

sed -i '' 's/REGISTER_NODE(Erode, "TOP\/Filter"/REGISTER_NODE(Erode, "Erode", "TOP\/Filter"/g' nodes/TOP/erode/erode_node.mm

sed -i '' 's/REGISTER_NODE(Math, "TOP\/Math"/REGISTER_NODE(Math, "Math", "TOP\/Math"/g' nodes/TOP/Math/math_node.mm

sed -i '' 's/REGISTER_NODE(Composite, "TOP\/Composite"/REGISTER_NODE(Composite, "Composite", "TOP\/Composite"/g' nodes/TOP/Composite/composite_node.mm

sed -i '' 's/REGISTER_NODE(Difference, "TOP\/Composite"/REGISTER_NODE(Difference, "Difference", "TOP\/Composite"/g' nodes/TOP/difference/difference_node.mm

sed -i '' 's/REGISTER_NODE(Threshold, "TOP\/Filter"/REGISTER_NODE(Threshold, "Threshold", "TOP\/Filter"/g' nodes/TOP/threshold/threshold_node.mm

sed -i '' 's/REGISTER_NODE(Resolution, "TOP\/Transform"/REGISTER_NODE(Resolution, "Resolution", "TOP\/Transform"/g' nodes/TOP/resolution/resolution_node.mm
sed -i '' 's/REGISTER_NODE(Resize, "TOP\/Transform"/REGISTER_NODE(Resolution, "Resolution", "TOP\/Transform"/g' nodes/TOP/resolution/resolution_node.mm

sed -i '' 's/REGISTER_NODE(Null, "TOP\/Utility"/REGISTER_NODE(Null, "Null", "TOP\/Misc"/g' nodes/TOP/null/null_node.mm

sed -i '' 's/REGISTER_NODE(Constant, "TOP\/Generator"/REGISTER_NODE(Constant, "Constant", "TOP\/Generator"/g' nodes/TOP/constant/constant_node.mm

sed -i '' 's/REGISTER_NODE(Blur, "TOP\/Filter"/REGISTER_NODE(Blur, "Blur", "TOP\/Filter"/g' nodes/TOP/blur/blur_node.mm

sed -i '' 's/REGISTER_NODE(Noise, "TOP\/Generator"/REGISTER_NODE(Noise, "Noise", "TOP\/Generator"/g' nodes/TOP/noise/noise_node.mm

sed -i '' 's/REGISTER_NODE(Switch, "TOP\/Utility"/REGISTER_NODE(Switch, "Switch", "TOP\/Misc"/g' nodes/TOP/switch/switch_node.mm

sed -i '' 's/REGISTER_NODE(Analyze, "TOP\/Analysis"/REGISTER_NODE(Analyze, "Analyze", "TOP\/Analysis"/g' nodes/TOP/analyze/analyze_node.mm

sed -i '' 's/REGISTER_NODE(Morphology, "TOP\/Filter"/REGISTER_NODE(Morphology, "Morphology", "TOP\/Filter"/g' nodes/TOP/morphology/morphology_node.mm

sed -i '' 's/REGISTER_NODE(Ramp, "TOP\/Generator"/REGISTER_NODE(Ramp, "Ramp", "TOP\/Generator"/g' nodes/TOP/ramp/ramp_node.mm

sed -i '' 's/REGISTER_NODE(Level, "TOP\/Color"/REGISTER_NODE(Level, "Level", "TOP\/Filter"/g' nodes/TOP/level/level_node.mm

sed -i '' 's/REGISTER_NODE(Glow, "TOP\/Style"/REGISTER_NODE(Glow, "Glow", "TOP\/Filter"/g' nodes/TOP/glow/glow_node.mm

sed -i '' 's/REGISTER_NODE(Dilate, "TOP\/Filter"/REGISTER_NODE(Dilate, "Dilate", "TOP\/Filter"/g' nodes/TOP/dilate/dilate_node.mm

sed -i '' 's/REGISTER_NODE(OpticalFlow, "TOP\/Analysis"/REGISTER_NODE(OpticalFlow, "Optical Flow", "TOP\/Analysis"/g' nodes/TOP/optical_flow/optical_flow_node.mm

sed -i '' 's/REGISTER_NODE(Transform, "TOP\/Transform"/REGISTER_NODE(Transform, "Transform", "TOP\/Transform"/g' nodes/TOP/Transform/transform_node.mm

sed -i '' 's/REGISTER_NODE(HSVAdjust, "TOP\/Color"/REGISTER_NODE(HSVAdjust, "HSV Adjust", "TOP\/Filter"/g' nodes/TOP/hsv_adjust/hsv_adjust_node.mm

# CHOP Nodes
sed -i '' 's/REGISTER_NODE(Sine, "CHOP\/Math"/REGISTER_NODE(Sine, "Sine", "CHOP\/Generator"/g' nodes/CHOP/math/sine_node.mm
sed -i '' 's/REGISTER_NODE(Add, "CHOP\/Math"/REGISTER_NODE(Add, "Add", "CHOP\/Math"/g' nodes/CHOP/math/add_node.mm
sed -i '' 's/REGISTER_NODE(Multiply, "CHOP\/Math"/REGISTER_NODE(Multiply, "Multiply", "CHOP\/Math"/g' nodes/CHOP/math/multiply_node.mm
sed -i '' 's/REGISTER_NODE(Time, "CHOP\/Generator"/REGISTER_NODE(Time, "Time", "CHOP\/Generator"/g' nodes/CHOP/generator/time_node.mm
