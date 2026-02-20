// Dear ImGui: standalone example application for OSX + Metal.
// Based on official example: https://github.com/ocornut/imgui/tree/master/examples/example_apple_metal

#import <Foundation/Foundation.h>
#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#include "../third_party/imgui/imconfig.h"
#include "../third_party/imgui/imgui.h"
#include "../third_party/imgui/backends/imgui_impl_metal.h"
#include "../third_party/imgui/backends/imgui_impl_osx.h"
#include "../third_party/imnodes/imnodes.h"
#include "node_editor.h"

//-----------------------------------------------------------------------------------
// AppViewController
//-----------------------------------------------------------------------------------

@interface AppViewController : NSViewController<NSWindowDelegate, MTKViewDelegate>
@property (nonatomic, readonly) MTKView *mtkView;
@property (nonatomic, strong) id <MTLDevice> device;
@property (nonatomic, strong) id <MTLCommandQueue> commandQueue;
@end

@implementation AppViewController

-(instancetype)initWithNibName:(nullable NSString *)nibNameOrNil bundle:(nullable NSBundle *)nibBundleOrNil
{
    self = [super initWithNibName:nibNameOrNil bundle:nibBundleOrNil];

    _device = MTLCreateSystemDefaultDevice();
    _commandQueue = [_device newCommandQueue];

    if (!self.device)
    {
        NSLog(@"Metal is not supported");
        abort();
    }

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;         // Enable Docking
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;       // Enable Multi-Viewport / Platform Windows

    // Setup ImNodes
    ImNodes::CreateContext();
    example::NodeEditorInitialize();

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    // ImGui::StyleColorsClassic();
    ImNodes::StyleColorsDark();

    // Setup Renderer backend
    ImGui_ImplMetal_Init(_device);

    return self;
}

-(MTKView *)mtkView
{
    return (MTKView *)self.view;
}

-(void)loadView
{
    self.view = [[MTKView alloc] initWithFrame:CGRectMake(0, 0, 1280, 720)];
}

-(void)viewDidLoad
{
    [super viewDidLoad];

    self.mtkView.device = self.device;
    self.mtkView.delegate = self;
    
    // 높은 프레임 레이트 설정 (120fps 또는 디스플레이 최대 주기)
    // 0으로 설정하면 디스플레이의 최대 새로고침 주기 사용
    self.mtkView.preferredFramesPerSecond = 120; // 120fps로 설정 (또는 0으로 디스플레이 최대 주기)

    ImGui_ImplOSX_Init(self.view);
    
    // NodeEditor에 Metal device 전달
    example::NodeEditorSetDevice(self.device);
    
    [NSApp activateIgnoringOtherApps:YES];
}

-(void)drawInMTKView:(MTKView*)view
{
    ImGuiIO& io = ImGui::GetIO();
    
    id<MTLCommandBuffer> commandBuffer = [self.commandQueue commandBuffer];

    MTLRenderPassDescriptor* renderPassDescriptor = view.currentRenderPassDescriptor;
    if (renderPassDescriptor == nil)
    {
        [commandBuffer commit];
        return;
    }
    
    // 실제 렌더 패스 텍스처 크기를 사용하여 DisplaySize 설정
    // 이렇게 하면 scissor rect가 render pass 크기와 항상 일치합니다
    id<MTLTexture> renderTexture = renderPassDescriptor.colorAttachments[0].texture;
    if (renderTexture != nil)
    {
        CGFloat framebufferScale = view.window.screen.backingScaleFactor ?: NSScreen.mainScreen.backingScaleFactor;
        
        // 실제 framebuffer (텍스처) 크기
        NSUInteger actualWidth = renderTexture.width;
        NSUInteger actualHeight = renderTexture.height;
        
        // DisplaySize는 논리적 크기 (실제 크기 / scale)
        io.DisplaySize.x = actualWidth / framebufferScale;
        io.DisplaySize.y = actualHeight / framebufferScale;
        io.DisplayFramebufferScale = ImVec2(framebufferScale, framebufferScale);
    }
    else
    {
        // fallback: view bounds 사용
        io.DisplaySize.x = view.bounds.size.width;
        io.DisplaySize.y = view.bounds.size.height;
        
        CGFloat framebufferScale = view.window.screen.backingScaleFactor ?: NSScreen.mainScreen.backingScaleFactor;
        io.DisplayFramebufferScale = ImVec2(framebufferScale, framebufferScale);
    }

    // Start the Dear ImGui frame
    ImGui_ImplMetal_NewFrame(renderPassDescriptor);
    ImGui_ImplOSX_NewFrame(view);
    ImGui::NewFrame();

    example::NodeEditorShow();

    // Rendering
    ImGui::Render();
    ImDrawData* draw_data = ImGui::GetDrawData();

    renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.45, 0.55, 0.60, 1.00);
    id <MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
    [renderEncoder pushDebugGroup:@"Dear ImGui rendering"];
    ImGui_ImplMetal_RenderDrawData(draw_data, commandBuffer, renderEncoder);
    [renderEncoder popDebugGroup];
    [renderEncoder endEncoding];

    // Update and Render additional Platform Windows
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
    }

    // Present
    [commandBuffer presentDrawable:view.currentDrawable];
    [commandBuffer commit];
}

-(void)mtkView:(MTKView*)view drawableSizeWillChange:(CGSize)size
{
    // MTKView의 drawable 크기가 변경될 때 호출됨
    // 다음 프레임에서 올바른 크기로 렌더링됨
    // 별도로 할 작업은 없지만, 로깅이 필요하면 여기에 추가
}

- (void)viewWillAppear
{
    [super viewWillAppear];
    self.view.window.delegate = self;
}

- (void)windowWillClose:(NSNotification *)notification
{
    example::NodeEditorShutdown();
    ImNodes::DestroyContext();
    ImGui_ImplMetal_Shutdown();
    ImGui_ImplOSX_Shutdown();
    ImGui::DestroyContext();
}

@end

//-----------------------------------------------------------------------------------
// AppDelegate
//-----------------------------------------------------------------------------------

@interface AppDelegate : NSObject <NSApplicationDelegate>
@property (nonatomic, strong) NSWindow *window;
@end

@implementation AppDelegate

-(BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender
{
    return YES;
}

-(instancetype)init
{
    if (self = [super init])
    {
        NSViewController *rootViewController = [[AppViewController alloc] initWithNibName:nil bundle:nil];
        self.window = [[NSWindow alloc] initWithContentRect:NSZeroRect
                                                  styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable | NSWindowStyleMaskMiniaturizable
                                                    backing:NSBackingStoreBuffered
                                                      defer:NO];
        self.window.contentViewController = rootViewController;
        [self.window setTitle:@"Ready Designer One"];
        [self.window center];
        [self.window makeKeyAndOrderFront:self];
    }
    return self;
}

@end

//-----------------------------------------------------------------------------------
// Application main() function
//-----------------------------------------------------------------------------------

int main(int, const char**)
{
    @autoreleasepool
    {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

        AppDelegate *appDelegate = [[AppDelegate alloc] init];   // creates window
        [NSApp setDelegate:appDelegate];

        [NSApp activateIgnoringOtherApps:YES];
        [NSApp run];
    }
    return 0;
}
