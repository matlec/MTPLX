// See ane_client.h. Adapted from Splash (Apache-2.0); see NOTICE.
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#include "ane_client.h"
#include <dlfcn.h>
#include <objc/runtime.h>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>

@protocol MtplxAneModel
+ (id)modelAtURL:(NSURL*)url key:(NSString*)key;
- (NSDictionary*)modelAttributes;
@end
@protocol MtplxAneClient
+ (id)sharedConnection;
- (BOOL)compileModel:(id)model options:(NSDictionary*)options qos:(unsigned)qos error:(NSError**)error;
- (BOOL)compiledModelExistsFor:(id)model;
- (void)purgeCompiledModel:(id)model;
- (BOOL)loadModel:(id)model options:(NSDictionary*)options qos:(unsigned)qos error:(NSError**)error;
- (BOOL)unloadModel:(id)model options:(NSDictionary*)options qos:(unsigned)qos error:(NSError**)error;
- (BOOL)evaluateWithModel:(id)model options:(NSDictionary*)options request:(id)request qos:(unsigned)qos error:(NSError**)error;
@end
@protocol MtplxAneSurface
+ (id)objectWithIOSurface:(IOSurfaceRef)surface;
@end
@protocol MtplxAneRequest
+ (id)requestWithInputs:(NSArray*)inputs inputIndices:(NSArray*)inputIndices outputs:(NSArray*)outputs
          outputIndices:(NSArray*)outputIndices weightsBuffer:(id)weights perfStats:(id)stats
         procedureIndex:(NSNumber*)procedure sharedEvents:(id)events transactionHandle:(NSNumber*)transaction;
- (void)setCompletionHandler:(void (^)(BOOL success, NSError* error))handler;
@end
@protocol MtplxAneEvents
+ (id)waitEventWithValue:(uint64_t)value sharedEvent:(id)event eventType:(uint64_t)type;
+ (id)signalEventWithValue:(uint64_t)value symbolIndex:(unsigned)index eventType:(int64_t)type sharedEvent:(id)event;
+ (id)sharedEventsWithSignalEvents:(NSArray*)signals waitEvents:(NSArray*)waits;
@end

namespace mtplx_ane {
namespace {
constexpr unsigned kQos = QOS_CLASS_USER_INTERACTIVE;
struct Api {
  bool ok = false;
  std::string why;
  id client = nil;
  Class model = nil, surface = nil, request = nil, events = nil, signal = nil, wait = nil;
};
const Api& api() {
  static Api* a = [] {
    auto* r = new Api();
    if (!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW)) {
      r->why = "AppleNeuralEngine does not load";
      return r;
    }
    r->model = NSClassFromString(@"_ANEModel");
    r->surface = NSClassFromString(@"_ANEIOSurfaceObject");
    r->request = NSClassFromString(@"_ANERequest");
    r->events = NSClassFromString(@"_ANESharedEvents");
    r->signal = NSClassFromString(@"_ANESharedSignalEvent");
    r->wait = NSClassFromString(@"_ANESharedWaitEvent");
    Class client = NSClassFromString(@"_ANEClient");
    if (!r->model || !r->surface || !r->request || !r->events || !r->signal || !r->wait || !client) {
      r->why = "AppleNeuralEngine lacks a class this client uses";
      return r;
    }
    const struct { Class c; SEL s; bool cls; } need[] = {
        {r->model, @selector(modelAtURL:key:), true},
        {client, @selector(sharedConnection), true},
        {client, @selector(compileModel:options:qos:error:), false},
        {client, @selector(compiledModelExistsFor:), false},
        {client, @selector(loadModel:options:qos:error:), false},
        {client, @selector(evaluateWithModel:options:request:qos:error:), false},
        {r->surface, @selector(objectWithIOSurface:), true},
        {r->request, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:), true},
        {r->signal, @selector(signalEventWithValue:symbolIndex:eventType:sharedEvent:), true},
        {r->wait, @selector(waitEventWithValue:sharedEvent:eventType:), true},
        {r->events, @selector(sharedEventsWithSignalEvents:waitEvents:), true},
    };
    for (auto& n : need) {
      if (!(n.cls ? class_getClassMethod(n.c, n.s) : class_getInstanceMethod(n.c, n.s))) {
        r->why = std::string("AppleNeuralEngine lacks ") + sel_getName(n.s);
        return r;
      }
    }
    r->client = [(Class<MtplxAneClient>)client sharedConnection];
    r->ok = r->client != nil;
    if (!r->ok) r->why = "no ANE shared connection";
    return r;
  }();
  return *a;
}
uint64_t fnv(const uint8_t* p, size_t n, uint64_t h = 1469598103934665603ull) {
  for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
  return h;
}
}  // namespace

std::string unavailable() { return api().ok ? "" : api().why; }

Surface::~Surface() {
  if (surface) CFRelease((IOSurfaceRef)surface);
}
std::string Surface::mil_strides() const {
  const uint64_t st = stride / esize, plane = uint64_t(rows) * st;
  return "[" + std::to_string(plane) + ", " + std::to_string(plane) + ", " + std::to_string(st) + ", 1]";
}
std::string Surface::mil_type(uint32_t w) const {
  return std::string("tensor_buffer<") + (int8 ? "int8" : "fp16") + ", shape=[1, 1, " + std::to_string(rows) + ", " +
         std::to_string(w ? w : width) + "], strides=" + mil_strides() + ", interleave_factors=[1, 1, 1, 1]>";
}
std::unique_ptr<Surface> make_surface(uint32_t rows, uint32_t width, bool int8) {
  const uint32_t es = int8 ? 1 : 2, stride = (width * es + 63) / 64 * 64;
  const uint64_t size = (uint64_t(stride) * rows + 16383) / 16384 * 16384;
  NSDictionary* p = @{(id)kIOSurfaceWidth : @(width), (id)kIOSurfaceHeight : @(rows), (id)kIOSurfaceBytesPerElement : @(es),
                      (id)kIOSurfaceBytesPerRow : @(stride), (id)kIOSurfaceAllocSize : @(size),
                      (id)kIOSurfacePixelFormat : @(int8 ? 0x4c303038 : 0x4c303068)};
  IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)p);
  if (!s) throw std::runtime_error("IOSurface creation failed");
  if (IOSurfaceGetBytesPerRow(s) != stride || IOSurfaceGetAllocSize(s) != size) {
    CFRelease(s);
    throw std::runtime_error("IOSurface took another row stride or size");
  }
  auto r = std::make_unique<Surface>();
  r->surface = (void*)s;
  r->base = IOSurfaceGetBaseAddress(s);
  r->rows = rows; r->width = width; r->esize = es; r->stride = stride; r->size = size; r->int8 = int8;
  return r;
}

struct Program::Impl {
  id model = nil;
  id<MTLSharedEvent> event = nil;
  uint64_t value = 0;
  std::mutex mutex;
  struct Proc { std::string function; std::vector<std::string> inputs; NSArray* inputIdx; NSArray* outputIdx; };
  std::vector<Proc> procs;
};

Program::Program(const std::string& mil, const std::vector<uint8_t>& weights) : impl_(new Impl) {
  const Api& a = api();
  if (!a.ok) throw std::runtime_error("ANE unavailable: " + a.why);
  @autoreleasepool {
    const uint64_t h = fnv(weights.data(), weights.size(), fnv((const uint8_t*)mil.data(), mil.size()));
    char key[32];
    snprintf(key, sizeof key, "%016llx", (unsigned long long)h);
    std::filesystem::path dir = std::filesystem::path(NSTemporaryDirectory().UTF8String) / "mtplx-ane" / key;
    std::filesystem::create_directories(dir);
    {
      std::ofstream(dir / "model.mil", std::ios::binary) << mil;
      std::ofstream f(dir / "weights.bin", std::ios::binary);
      f.write((const char*)weights.data(), weights.size());
    }
    impl_->model = [(Class<MtplxAneModel>)a.model modelAtURL:[NSURL fileURLWithPath:@(dir.c_str()) isDirectory:YES] key:@(key)];
    if (!impl_->model) throw std::runtime_error("ANE model creation failed");
    NSError* err = nil;
    auto t0 = std::chrono::steady_clock::now();
    if (![a.client compiledModelExistsFor:impl_->model]) {
      if (![a.client compileModel:impl_->model options:@{@"kANEFModelType" : @"kANEFModelMIL", @"kANEFNetPlistFilenameKey" : @"model.mil"} qos:kQos error:&err])
        throw std::runtime_error(std::string("ANE compile failed: ") + (err ? err.description.UTF8String : "?"));
    }
    if (![a.client loadModel:impl_->model options:@{} qos:kQos error:&err]) {
      [a.client purgeCompiledModel:impl_->model];
      err = nil;
      if (![a.client compileModel:impl_->model options:@{@"kANEFModelType" : @"kANEFModelMIL", @"kANEFNetPlistFilenameKey" : @"model.mil"} qos:kQos error:&err] ||
          ![a.client loadModel:impl_->model options:@{} qos:kQos error:&err])
        throw std::runtime_error(std::string("ANE load failed: ") + (err ? err.description.UTF8String : "?"));
    }
    compile_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    NSDictionary* desc = [impl_->model modelAttributes][@"ANEFModelDescription"];
    NSArray* symbols = desc[@"kANEFModelInputSymbolsArrayKey"];
    NSDictionary* names = desc[@"kANEFModelProcedureNameToIDMapKey"];
    NSArray* entries = desc[@"ANEFModelProcedures"];
    if (![symbols isKindOfClass:NSArray.class] || ![names isKindOfClass:NSDictionary.class] || ![entries isKindOfClass:NSArray.class])
      throw std::runtime_error("ANE program does not describe its procedures");
    impl_->procs.resize(entries.count);
    for (NSString* fn in names) impl_->procs[[names[fn] unsignedIntegerValue]].function = fn.UTF8String;
    for (NSDictionary* e in entries) {
      auto& p = impl_->procs[[e[@"ANEFModelProcedureID"] unsignedIntegerValue]];
      p.inputIdx = e[@"ANEFModelInputSymbolIndexArray"];
      p.outputIdx = e[@"ANEFModelOutputSymbolIndexArray"];
      for (NSNumber* i in p.inputIdx) p.inputs.push_back([symbols[i.unsignedIntValue] UTF8String]);
    }
    impl_->event = [MTLCreateSystemDefaultDevice() newSharedEvent];
  }
}
Program::~Program() {
  @autoreleasepool {
    if (impl_ && impl_->model) {
      NSError* err = nil;
      [api().client unloadModel:impl_->model options:@{} qos:kQos error:&err];
    }
  }
}
int Program::procedure(const std::string& function) const {
  for (size_t i = 0; i < impl_->procs.size(); ++i)
    if (impl_->procs[i].function == function) return int(i);
  throw std::runtime_error("ANE program has no function " + function);
}
void Program::run(int procedure, const std::map<std::string, Surface*>& inputs, Surface* output) {
  std::lock_guard lock(impl_->mutex);
  const Api& a = api();
  @autoreleasepool {
    const auto& p = impl_->procs.at(procedure);
    NSMutableArray* objs = [NSMutableArray arrayWithCapacity:p.inputs.size()];
    for (const auto& name : p.inputs) {
      auto it = inputs.find(name);
      if (it == inputs.end()) throw std::runtime_error("ANE run lacks input " + name);
      [objs addObject:[(Class<MtplxAneSurface>)a.surface objectWithIOSurface:(IOSurfaceRef)it->second->surface]];
    }
    id out = [(Class<MtplxAneSurface>)a.surface objectWithIOSurface:(IOSurfaceRef)output->surface];
    const uint64_t v = impl_->value;
    impl_->value += 2;
    id evs = [(Class<MtplxAneEvents>)a.events
        sharedEventsWithSignalEvents:@[ [(Class<MtplxAneEvents>)a.signal signalEventWithValue:v + 2 symbolIndex:0 eventType:0 sharedEvent:impl_->event] ]
                          waitEvents:@[ [(Class<MtplxAneEvents>)a.wait waitEventWithValue:v + 1 sharedEvent:impl_->event eventType:0] ]];
    id req = [(Class<MtplxAneRequest>)a.request requestWithInputs:objs inputIndices:p.inputIdx outputs:@[ out ] outputIndices:p.outputIdx
                                                  weightsBuffer:nil perfStats:nil procedureIndex:@(procedure) sharedEvents:evs transactionHandle:nil];
    __block BOOL ok = NO;
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    [req setCompletionHandler:^(BOOL success, NSError*) { ok = success; dispatch_semaphore_signal(sem); }];
    NSError* err = nil;
    if (![a.client evaluateWithModel:impl_->model options:@{} request:req qos:kQos error:&err])
      throw std::runtime_error(std::string("ANE evaluation failed: ") + (err ? err.description.UTF8String : "?"));
    impl_->event.signaledValue = v + 1;
    if (dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, 20ll * NSEC_PER_SEC)) != 0)
      throw std::runtime_error("ANE evaluation did not report within 20 s");
    if (!ok) throw std::runtime_error("ANE evaluation reported failure");
  }
}
Program* dummy_for_link = nullptr;
}  // namespace mtplx_ane
