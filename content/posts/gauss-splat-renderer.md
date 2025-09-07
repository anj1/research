+++
title = 'Hacking Together the Fastest WebGL2 Gaussian Splat Renderer'
date = 2025-09-07T01:38:00-00:00
author = "A Nejati"
draft = false
math = true
+++

[Gaussian Splats](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) are becoming really popular for computer graphics. You can do photorealistic rendering in a way that wasn't really possible before and you can use a lot of machine learning methods for things like learning scenes and generating new scenes. But compared to traditional (polygon) based rendering, there's a tradeoff: you need to process, sort, and render a lot more primitives (millions of splats) in real time.

The official papers usually show off CUDA-based renderers, which is great but locks you into specific hardware or requires native apps that can't run in a browser. I wanted to build something that could hit 60fps on basically any machine with a modern browser. That meant figuring out how to do the heaviest part (the sorting) entirely on the GPU using web-standard APIs.

So a while ago, [I built a WebGL2 renderer](https://github.com/PanverseRobotics/portality-web-viewer) that does exactly that. The core idea is to offload the entire per-frame sort of millions of splats to the GPU using a bitonic sorting network implemented in fragment shaders.

Here’s the breakdown of the tech stack and the main tricks I used:

1.  **The Platform:** Why WebGL2 was the right tool for the job over WebGPU or WebGL1.
2.  **The Geometry:** Ditching complex geometry and treating each splat as a single vertex to abuse the GPU's rasterizer.
3.  **The Sort:** A deep dive into the bitonic sort implementation which is the heart of the renderer.
4.  **The Hacks:** Using WebGL2 features like Multiple Render Targets to efficiently shuffle data around.

Let's get into the platform choice first.

## WebGL2

When you're building for the web, you've got a few graphics APIs to choose from: WebGL and WebGL2, for example, and also a more compute-oriented API which is WebGPU.

WebGPU is obviously the future. It has proper compute shaders which would make a GPGPU task like this much more straightforward to implement. But I was building a product, and browser support for WebGPU just wasn't (and still isn't quite) there yet. I needed something that would run everywhere, period. WebGL1 has the best support but it's based on OpenGL ES 2.0 and is missing fundamental features for compute work.

So, **WebGL2 was the only practical choice**. It's got 97%+ browser support and is based on OpenGL ES 3.0, which added just enough GPGPU-friendly features to make this project possible.

### GPGPU with a graphics library

The main GPGPU pattern in WebGL is to dump all your data into textures and then run fragment shaders to process it, rendering the output to another texture.

WebGL1 makes this painful because it mostly just supports 8-bit-per-channel textures (like `RGBA8`). To store a 32-bit float, you have to manually encode it across the four 8-bit channels of a single pixel, then decode it back in the shader. Slow.

WebGL2 fixes this with native support for better texture formats:

* **Float Textures:** You can just create a texture with 32-bit float channels (`R32F`, `RGBA32F`). This is essential. I use these to store all the splat data: positions, colors, covariance matrices.
* **Integer Textures:** Just as important is support for 32-bit unsigned integer textures (`R32UI`, `RG32UI`). A sorting algorithm is all about shuffling indices. My sorting kernels use `usampler2D` to read these index maps directly.

### GLSL 3.00 ES

With WebGL2 you also get support for GLSL 3.00 ES shader language, which adds some important features for compute:

* **`texelFetch`:** This function lets you read from a texture using integer coordinates (e.g., `texelFetch(myTexture, ivec2(10, 5), 0)`). This is how you do array-style lookups. In contrast, the old `texture2D` function uses normalized `[0, 1]` coordinates and is subject to filtering, which you absolutely do not want when manipulating raw data.
* **Bitwise Operators:** My bitonic sort implementation is built on bitwise operations (`^` for XOR, `&` for AND). You can see it in the kernel code: `uint l = idx ^ uStepj;`. This is a core part of the sorting logic. These operators are native in GLSL 3.00 ES, making the implementation clean and fast.
* **Multiple Render Targets (MRTs):** This is the killer feature for efficiency. MRTs let you write to multiple textures in a single shader pass. Once I've sorted the splats and have a texture full of the new sorted indices, I use MRTs to reorder all the splat attribute textures (positions, colors, etc.) in one go. We'll explore in detail later.

Basically, WebGL2 gave me a stable, cross-browser platform with just enough raw power to implement a complex, high-performance sorting algorithm. It let me treat the GPU like the massively parallel data processor it is.

## The Geometry Hack: One Point, One Splat

So we've established that all our splat data, which is position, rotation (as a quaternion or covariance matrix), scale, color, and opacity, lives in textures on the GPU. How do we tell the GPU to draw them?

The naive approach might be to create a quad (two triangles, 4-6 vertices) for every single splat. For a few thousand objects, that's fine. For millions of splats, it's less ideal. Pushing 4 million vertices through the pipeline just to draw 1 million splats is pretty hard on the gpu.

The solution is to slightly abuse the **`gl.POINTS`** primitive type. Instead of sending triangles or lines, we just send a giant list of single vertices. Each vertex represents the 3D center of a Gaussian splat. For a scene with 2 million splats, we create a vertex buffer with exactly 2 million positions, not 8 million.

Now you might say: A point is just a point. A single pixel. You're right. We need to expand it into a 2D shape that covers the splat's projected area on the screen.

In native OpenGL or DirectX, you'd typically solve this with a geometry shader. A geometry shader can take a primitive like a point and programmatically output more complex geometry, like a quad, on the fly. Unfortunately, **WebGL2 does not have geometry shaders.** So we need to find some other way to do this.

Here's the hack. We can use a feature called "point sprites" to get the job done using only a vertex and fragment shader.

1.  **The Vertex Shader:** In the vertex shader, we do the standard 3D projection to get the splat's center position on the screen. But we also do some extra math. Based on the splat's 3D covariance and its distance from the camera, we calculate how big it *should* appear on the screen in pixels. We then write this pixel size to a special built-in variable: **`gl_PointSize`**.

2.  **The Rasterizer:** When the GPU's rasterizer sees `gl_PointSize`, it doesn't just draw a single pixel. It creates a screen-aligned square of fragments centered on the vertex's projected position, with a width and height equal to `gl_PointSize`. So, we've effectively expanded our point into a quad without a geometry shader.

3.  **The Fragment Shader:** Now, for every fragment (pixel) inside that square, the fragment shader runs. How do we know which part of the splat we're drawing? The GPU gives us another variable: **`gl_PointCoord`**. This is a `vec2` that gives us the local coordinate within the point sprite, ranging from `(0,0)` at the top-left to `(1,1)` at the bottom-right.

Using `gl_PointCoord`, we can treat our square point sprite just like a regular quad. We can calculate the 2D Gaussian function based on the coordinate. This lets us shape the splat into an ellipse and give it that soft, fuzzy look. We can also use `discard` to throw away any fragments that fall outside the ellipse, so we're not just drawing a bunch of colored squares.

This vertex-to-fragment-shader pipeline achieves what we need. We get the memory efficiency of sending just one vertex per splat, and we get the flexibility to shape and color it correctly in the fragment shader. All our splat attributes (read from textures using the vertex's ID, `gl_VertexID`) are simply passed from the vertex shader to the fragment shader as `varying`s.

This setup is the basic way the renderer works. But there's another issue: splats are transparent. To look right, they *must* be rendered in the correct back-to-front order. And for that, we need to sort them. Every. Single. Frame. And sorting is expensive.

## The Main Event: Full-Scene GPU Bitonic Sort

When I was writing my renderer, most other web renderers sorted the splats on the CPU. But even a highly optimized `Array.prototype.sort()` would kill performance. The round trip of pulling millions of depth values from the GPU, sorting on the CPU, and pushing the results back is a massive bottleneck that would destroy our frame rate.The original gauss splat code sorted on the GPU, but it used custom CUDA functions which we don't have access to. The big challenge is sorting on the gpu using webgl2.

Standard sorting algorithms like Quicksort or Mergesort aren't a great fit for GPUs. They rely on recursion and data-dependent branching, which is not a good fit for the GPU's architecture. GPUs want all their cores to be doing the same thing at a time. This is where **Bitonic Sort** comes in. It’s what's known as a "sorting network." This means the entire sequence of comparisons and swaps is fixed and data-independent. It doesn't matter if the data is already sorted, reversed, or random; the exact same set of operations will be performed. This predictable, branch-free nature makes it a perfect match for the GPU. 

Its time complexity on a serial machine is $O(n \log^2 n)$, which is worse than the $O(n \log n)$ of Mergesort, but that doesn't matter. Because every comparison in a given stage can be run in parallel, its performance on the GPU is quite good. In the next section I'll go into some technical detail on how it works.

### The Bitonic Kernel

The implementation lives in a fragment shader, our "sorting kernel." The core idea is to treat our giant 1D list of splats as a 2D texture. The JS code on the CPU acts as a simple driver, running a loop that executes the sort stages in order.

For each stage, we run our sorting kernel. The kernel's job is simple:

1.  Each pixel in our texture represents one splat.
2.  The JS driver passes two uniforms, `uStepj` and `uStepk`, which define the current stage of the sort. These values determine how "far away" each element's comparison partner is.
3.  Inside the shader, each pixel uses its coordinate (`gl_FragCoord`) and the uniforms to calculate the coordinate of the other splat it needs to compare itself against. This is done with simple bitwise logic: `lx = idxx ^ uStepjx`.
4.  It fetches its own value and its partner's value from an input texture.
5.  It compares them.
6.  Based on the comparison, it writes the correct value (either its own or its partner's, depending on whether it should keep the smaller or larger one) to an **output texture**.

### Ping-Ponging Textures

You can't read from and write to the same texture in a single draw call. So, we use a "ping-pong" or "double-buffering" technique. We have two identical textures, let's call them `A` and `B`.

* **Pass 1:** We read from `A`, run the compare-and-swap logic, and write the results to `B`.
* **Pass 2:** We read from `B`, run the next stage of logic, and write the results back to `A`.

We continue this, swapping which texture is the source and which is the destination, for all $O(\log^2 n)$ stages of the sort. My code manages this by having an array of two kernels (`pipeline.bitKer`) and alternating between them.

### Sorting Keys, Not Payloads

It would be incredibly inefficient to swap all the splat data (position, color, covariance) around for every single comparison step. The bandwidth costs would be huge.

Instead, we use a key-value approach. At the start of the frame, I create a temporary `RG32UI` texture. For each splat, I store two things:

* **R Channel (Key):** The depth of the splat (its distance to the camera, packed into a 32-bit float and reinterpreted as an integer for sorting).
* **G Channel (Index):** The splat's original index (from 0 to N-1).

The bitonic sort kernel operates *only* on this lightweight texture, sorting the `(depth, index)` pairs based on the `depth` key.

The final output of this entire multi-pass sorting process isn't a texture with re-arranged splat data. It's a texture containing all the original indices, but now in the correct back-to-front order. This sorted index map is the critical piece of information we need for the final rendering pass. Now we just have to use it.

## The Final Trick: Permuting Data with MRTs

So, we've got our sorted index map. It's a texture where each pixel tells us the original index of the splat that *should* be drawn at this position in the sorted sequence. For example, the pixel at `(0,0)` might contain the index `1,234,567`, meaning the splat that was originally at index 1,234,567 is the furthest from the camera and should be drawn first.

How do we use this map?

The most straightforward way would be to pass this index map to our final rendering shader. In the vertex shader, for the splat we're about to draw (say, `gl_VertexID = 0`), we would first read the index map at coordinate `0` to get the real data index (`1,234,567`). Then, we'd use *that* index to go and fetch the splat's position, color, and covariance data from our main data textures.

This is called an **indirect** or **dependent texture read**. And it's a performance killer. The GPU has to wait for the result of the first texture read (from the index map) before it even knows the memory address of the data it needs for the second read. This creates stalls in the pipeline and kills the GPU's ability to hide memory latency. It works, but it's not fast.

### The Permutation Pass: A Better Way

To achieve maximum performance, we need to eliminate that indirect lookup in the final, hot rendering loop. The solution is to add one more step before rendering: a full **permutation pass**.

The idea is to create a brand new set of splat attribute textures that are already physically sorted. We'll have `sortedPositions`, `sortedColors`, `sortedCovariances`, and so on. The splat data at index `0` of these textures will be for the furthest splat, index `1` for the second furthest, and so on.

Once we have these sorted textures, the final rendering pass becomes dead simple and blazing fast. The vertex shader for `gl_VertexID` can just read directly from `sortedPositions[gl_VertexID]`, `sortedColors[gl_VertexID]`, etc. All memory access is linear and predictable, which is exactly what the GPU loves.

But a permutation pass sounds expensive. We have to read from the index map, then read from all our original attribute textures, and write out to a whole new set of attribute textures. Doing that sequentially would involve one pass to create `sortedPositions`, a second pass for `sortedColors`, and this would be slow.

### One Pass to Rule Them All: Multiple Render Targets (MRTs)

This is where that final WebGL2 superpower comes into play: **Multiple Render Targets (MRTs)**. MRTs allow a single fragment shader to write to several different output textures at the same time.

I created a single permutation kernel (`createTexturePermuteKernel` in the code) that does the following in one draw call:

1.  The kernel is set up to output to multiple textures simultaneously (`layout(location = 0) out vec4 outPosition; layout(location = 1) out vec4 outColor;`... etc.).
2.  For each pixel it's processing, it first does a single read from the sorted index map to find out which original splat's data it needs to grab.
3.  It then reads the position, color, and covariance for that splat from all the original (unsorted) attribute textures.
4.  Finally, it writes the position data to the first output texture, the color data to the second, the covariance to the third, and so on.

By using MRTs, we re-order all of our splat data in a single, efficient pass.

### The Full Pipeline

So, to recap, the entire process for rendering a single frame looks like this:

1.  **Depth Pass:** A quick pass to calculate the camera-space depth for every splat and store it in a texture.
2.  **Sort Pass:** Execute the bitonic sort network. This is a series of "ping-pong" draw calls that operate on a lightweight `(depth, index)` texture, producing our final sorted index map.
3.  **Permute Pass:** A single draw call using our MRT kernel to generate a new, fully-sorted set of splat attribute textures.
4.  **Render Pass:** The final draw call. We draw our `gl.POINTS` and the shaders read linearly from the sorted textures. No indirection, no complexity, just raw speed.

This combination of a GPU-native sorting network and an efficient MRT-based permutation pass is the engine that drives the renderer. By understanding the hardware and leveraging the right WebGL2 features, we can build a system that sorts and renders millions of primitives in real-time, right in your browser.