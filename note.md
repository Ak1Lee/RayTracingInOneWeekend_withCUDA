## CUDA 并行
## 要做什么


将Ray Tracing in One Weekdend 实现的程序用cuda作并行(作为课程大作业)

对每个像素点做采样。
```
//TODO 实现完了再补充
```

## CUDA虚函数实现

### 为什么是虚函数
假设我们已经开始对像素点采样，已经生成了`Ray`类型的`ray`对象，下一步要怎么做？
- 把要渲染的物体收集起来
- 对这些物体逐个对光线进行相交测试，返回结果并处理
- 最终得到这个像素的颜色
对于需要做相交测试的物体，可以让他们基于一个基类 `Hittable`类 派生而来，从而建立存储结构。在这个类上再派生出`Sphere`类 或者 `Cube`类
在`Hittable`类上生命一个虚函数`Hit(···,const& Ray ray,···)`传入光线进行相交测试，在`Sphere`上再实现`Sphere`的`Hit()`
同时我建立了一个名为`Hittable_list`的`Hittable*`数组，存了一些`Sphere*`对象指针，指向我的Sphere对象。
在我计算光追的时候，我只需要先准备`ray`，逐个调用 Hittable* 的`Hit`函数就可以完成对不同类的相交测试。

### 问题在哪？

困扰了我有点久 搜索解决方法的时候还找到了同届同校隔壁班竞赛佬的文章，可惜代码读不太明白，唉，智商

google一下
>"It is not allowed to pass as an argument to a global function an object of a class with virtual functions. "
>The reason is that if you instantiate the object on the host, then the virtual function table gets populated with host pointers. When you copy this object to the device, these host-pointers become meaningless.
>https://forums.developer.nvidia.com/t/can-cuda-properly-handle-pure-virtual-classes/37588/4

讲的比较清楚，简单来说，正常在host端实例化对象的话，虚函数表的指针指向的是host的内存，在device端处理，首先需要将对象copy到device的内存中，但是此时虚函数表的指针指向的地址仍然是host端的内存，所以此时无法正确调用虚函数。

### 如何解决？
让对象直接在device的内存上创建。

先写到这 歇歇


```
__global__ void create_function(
    Hittable** hittable,
    point3 center,
    float radius
    )
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {

        *hittable = new Sphere(center, radius);
    }
}
```

```
__global__ void delete_function(Hittable** hittable)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {

        delete *hittable;
    }
}
```

```
__device__ color ray_color_sample_test(const Ray& r, Hittable const* const* __restrict__ hittables, int depth)
{

    if ((*hittables)->hit(r, ray_t, rec)) 
    ...
```