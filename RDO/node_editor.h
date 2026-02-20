#pragma once

@protocol MTLDevice;

namespace example
{
void NodeEditorInitialize();
void NodeEditorSetDevice(id<MTLDevice> device);
void NodeEditorShow();
void NodeEditorShutdown();
} // namespace example
