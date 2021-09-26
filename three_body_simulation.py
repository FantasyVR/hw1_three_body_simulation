import taichi as ti
ti.init(arch=ti.gpu)
N, dim = 3, 3
x = ti.Vector.field(dim,ti.f32,shape=N)
v = ti.Vector.field(dim,ti.f32,shape=N)
a = ti.Vector.field(dim,ti.f32,shape=N)
f = ti.Vector.field(dim,ti.f32,shape=N)
m = ti.field(ti.f32,shape=N)
color = ti.Vector.field(3, float, N)
@ti.kernel 
def init():
    x[0] = ti.Vector([ -0.970004360000000, 0.243087530000000,0.0])
    x[1] = ti.Vector([ 0.970004360000000, -0.243087530000000,0.0])
    x[2] = ti.Vector([0.0,0.0,0.0])
    v[0] = ti.Vector([-0.466203685000000, -0.432365730000000, 0.0])
    v[1] = ti.Vector([-0.466203685000000, -0.432365730000000, 0.0])
    v[2] = ti.Vector([ 0.932407370000000,  0.864731460000000, 0.0])
    for i in range(N):
        m[i] = 1.0
    color[0] = ti.Vector([1.0,0.0,0.0])
    color[1] = ti.Vector([0.0,1.0,0.0])
    color[2] = ti.Vector([0.0,0.0,1.0])


@ti.kernel
def step(h: ti.f32):
    r01 = x[1] - x[0]
    r02 = x[2] - x[0]
    r12 = x[2] - x[1]
    d01 = r01.norm()
    d02 = r02.norm()
    d12 = r12.norm()

    #compute gravatation force
    G = 1.0
    f0 = G * m[0] * m[1] * r01 / d01**3
    f1 = G * m[0] * m[2] * r02 / d02**3
    f2 = G * m[1] * m[2] * r12 / d12**3

    # compute acc
    a[0] = ( f0 + f1)/m[0]
    a[1] = (-f0 + f2)/m[1]
    a[2] = (-f1 - f2)/m[2]

    for i in range(N):
        v[i] += a[i] * h
        x[i] += v[i] * h


init()
# display on CPU arch if you don't have GGUI
# gui = ti.GUI("Three body system")
# while gui.running:
#     step(0.005)
#     gui.circles(x.to_numpy(), radius=4.0, color=0xFF0000)
#     gui.show()

# display on GPU based on GGUI
window = ti.ui.Window('Three body system', (800, 800))
canvas = window.get_canvas()
scene =  ti.ui.Scene()
camera = ti.ui.make_camera()
camera.position(0.0,0.0,2.0)
camera.lookat(0.0, 0.0, 0.0)
camera.fov(80)
while window.running:
    step(0.001)
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((0, 0, 0))
    scene.particles(x, per_vertex_color=color, radius=0.1)
    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.5, 0.5, 0.5))
    canvas.scene(scene)
    window.show()