import numpy as np
import matplotlib.pyplot as plt

# authors: Pranav Rajan and Carlos Martinez
# starter code from https://gist.github.com/rossant/6046463

# parameters for the scene for rendering
w = 840
h = 680

# function for normalizing vectors
def normalize(x):
    x /= np.linalg.norm(x)
    return x


# plane intersection equation
def intersect_plane(O, D, P, N):
    denom = np.dot(D, N)
    if np.abs(denom) < 1e-6:
        return np.inf
    d = np.dot(P - O, N) / denom
    if d < 0:
        return np.inf
    return d

# parametric intersection for plane intersection (khan academy)
def intersection_point(O, t, D):
    if t == np.inf:
        return t

    return O + t * D

# sphere intersection equation
# computes the intersection of the points using the numerical way described by scratch a pixel for intersections
    # Return the distance from O to the intersection of the ray (O, D) with the
    # sphere (S, R), or +inf if there is no intersection.
    # O and S are 3D points, D (direction) is a normalized vector, R is a scalar.
def intersect_sphere(O, D, S, R):
    a = np.dot(D, D)
    OS = O - S
    b = 2 * np.dot(D, OS)
    c = np.dot(OS, OS) - R * R
    disc = b * b - 4 * a * c
    if disc > 0:
        distSqrt = np.sqrt(disc)
        q = (-b - distSqrt) / 2.0 if b < 0 else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return np.inf

    # Return the distance from O to the intersection of the ray (O, D) with the
    # triangle (A, B, C, N), or +inf if there is no intersection.
    # O, A, B and C are 3D points, D (direction) and N (normal) are normalized vectors.
def intersect_triangle(O, D, A, B, C, N):
    t = intersect_plane(O, D, A, N)
    intersectionPoint = intersection_point(O, t, D)
    if (np.isinf(intersectionPoint).all()):
        return np.inf

    v1 = normalize(A - B)
    if (np.dot(N, np.cross(v1, (normalize(intersectionPoint - B)))) < 0):
        return np.inf

    v2 = normalize(B - C)
    if (np.dot(N, np.cross(v2, (normalize(intersectionPoint - C)))) < 0):
        return np.inf

    v3 = normalize(C - A)
    if (np.dot(N, np.cross(v3, (normalize(intersectionPoint - A)))) < 0):
        return np.inf

    return t

 # intersection function for the different objects
def intersect(O, D, obj):
    if obj['type'] == 'plane':
        return intersect_plane(O, D, obj['position'], obj['normal'])
    elif obj['type'] == 'sphere':
        return intersect_sphere(O, D, obj['position'], obj['radius'])
    elif obj['type'] == 'triangle':
        return intersect_triangle(O, D, obj['A'], obj['B'], obj['C'], obj['normal'])

# function that gets the normal vector for the different objects
def get_normal(obj, M):
    # Find normal.
    if obj['type'] == 'sphere':
        N = normalize(M - obj['position'])
    elif obj['type'] == 'plane':
        N = obj['normal']
    elif obj['type'] == 'triangle':
        N = obj['normal']
    return N

# function that gets the color for the different objects
def get_color(obj, M):
    color = obj['color']
    if not hasattr(color, '__len__'):
        color = color(M)
    return color

# the trace function that traces the rays through the scene
def trace_ray(rayO, rayD):
 # find intersection point
    t = np.inf
    for i, obj in enumerate(scene):
        t_obj = intersect(rayO, rayD, obj)
        if t_obj < t:
            t, obj_idx = t_obj, i
    # return if ray does not intersect the object
    if t == np.inf:
        return
    # identify the object that was hit
    obj = scene[obj_idx]
    # Determine the hit point on the object
    M = rayO + rayD * t
    # Object Properties
    N = get_normal(obj, M)
    color = get_color(obj, M)

# normalize the intersection points
    toO = normalize(O - M)
   # compute the color
    col_ray = ambient
    for index, light in enumerate(light_array):
        toL = normalize(light - M)
        # shadow light intersection
        l = [intersect(M + N * .0001, toL, obj_sh)
             for k, obj_sh in enumerate(scene) if k != obj_idx]
        if l and min(l) < np.inf:
            continue
        # Lambertian shading
        col_ray += obj.get('diffuse_c', diffuse_c) * max(np.dot(N, toL), 0) * color
        # Blinn-Phong shading (specular).
        col_ray += obj.get('specular_c', specular_c) * max(np.dot(N, normalize(toL + toO)), 0) ** specular_k * \
                   color_light[index]

    return obj, M, N, col_ray

# function for adding spheres to the scene
def add_sphere(position, radius, color):
    return dict(type='sphere', position=np.array(position),
                radius=np.array(radius), color=np.array(color), reflection=.5)

# function for adding triangles to the scene
def add_triangle(A, B, C, color):
    vector1 = [B[0] - A[0], B[1] - A[1], B[2] - A[2]]
    vector2 = [C[0] - A[0], C[1] - A[1], C[2] - A[2]]
    return dict(type='triangle', A=np.array(A),
                B=np.array(B), C=np.array(C),
                color=np.array(color), reflection=.5, normal=np.array(np.cross(vector2, vector1)))

# function for adding triangle mesh the scene
def add_triangle_mesh(scene, A, B, C, D, E, color):
    scene.append(add_triangle(A, B, C, color))
    scene.append(add_triangle(B, D, C, color))
    scene.append(add_triangle(E, A, C, color))

# function for adding planes to the scene
def add_plane(position, normal):
    return dict(type='plane', position=np.array(position),
                normal=np.array(normal),
                color=lambda M: (color_plane0
                                 if (int(M[0] * 2) % 2) == (int(M[2] * 2) % 2) else color_plane1),
                diffuse_c=.75, specular_c=.5, reflection=.25)


# the scene
# colors for the checkerboard
color_plane0 = 1. * np.ones(3)
color_plane1 = 0. * np.ones(3)
# the bear scene and the triangle mesh
scene = [add_sphere([.75, .1, 1.], .6, [0., 0., 1.]),
         add_sphere([-2.75, .1, 3.5], .6, [1., .572, .184]),
         add_plane([0., -.5, 0.], [0., 1., 0.]),
         add_sphere([0., 0., 1], .9, [1., 0.5, 0.]),        # Nose Body
         add_sphere([0., .5, 2], 1.5, [1., 0.25, 0.]),      # Head
         add_sphere([0.5, 0.8, .5], .1, [0., 0., 0.]),      # Right Eyes
         add_sphere([-0.5, 0.8, .5], .1, [0., 0., 0.]),     # Left Eyes
         add_sphere([0., 0.5, .25], .2, [0., 0., 0.]),      # Nose
         add_sphere([0., -0.1, .25], .3, [0., 0., 0.]),     # Mouth
         add_sphere([1., 1.25, 1.25], .5, [1., 0.25, 0.]),  # Right Ear
         add_sphere([-1., 1.25, 1.25], .5, [1., 0.25, 0.]), # Left Ear
         add_sphere([1., 1.25, 1.], .3, [0., 0., 0.]),      # Inner Right Ear
         add_sphere([-1., 1.25, 1.], .3, [0., 0., 0.]),     # Inner Left Ear
         # add_sphere([0., .5, 9.], 8., [1., 1., 1.]),        # White Sky
         add_sphere([0., -0.65, 0.], .3, [1., 0.25, 0.]),   # Body
         add_sphere([.5, 0., 1], .8, [1., 0.5, 0.]),        # Right Cheek
         add_sphere([-.5, 0., 1], .8, [1., 0.5, 0.]),       # Left Cheek
         add_sphere([.6, -.9, 0], .6, [1., 0.25, 0.]),      # Right Shoulder
         add_sphere([-.6, -.9, 0], .6, [1., 0.25, 0.]),     # Left Shoulder
         add_plane([0., -.5, 0.], [0., 1., 0.])             # Plane
         ]
add_triangle_mesh(scene, [-1., -.2, 0], [-0.3, -.2, 0], [-0.65, .55, 0], [.3, .0, 0], [-1.4, .4, 0], [1., .0, .0])
add_triangle_mesh(scene, [1., .2, 0], [0.3, .2, 0], [0.65, -.55, 0], [.3, .0, 0], [1.4, -.4, 0], [0., 1.0, 0.])


# Light position and color.
light_array = [np.array([5., 5., -10.]), np.array([-5., 15., -10.]), np.array([15., 25., -10.])]
color_light = [np.ones(3), np.array([.5, .223, .5]), np.ones(3)]

# Constants for shading
ambient = .05
diffuse_c = 1.
specular_c = 1.
specular_k = 50

depth_max = 5  # the number of bounces for reflections
col = np.zeros(3)  # initializing the color
O = np.array([0., 0.35, -1.])  # Camera.
Q = np.array([0., 0., 0.])  # Camera direction
img = np.zeros((h, w, 3)) # initialize the camera

r = float(w) / h # constant for the screen space offset
# Screen coordinates: x0, y0, x1, y1.
S = (-1., -1. / r + .25, 1., 1. / r + .25)

# iterate over the x, y coordinates for the screen space
for i, x in enumerate(np.linspace(S[0], S[2], w)):
    if i % 10 == 0:
        print(i / float(w) * 100, "%")
    for j, y in enumerate(np.linspace(S[1], S[3], h)):
        col[:] = 0
        Q[:2] = (x, y)
        # the direction normalized
        D = normalize(Q - O)
        depth = 0
        rayO, rayD = O, D
        reflection = 1.
        # reflection calculation
        while depth < depth_max:
            traced = trace_ray(rayO, rayD)
            if not traced:
                break
            obj, M, N, col_ray = traced
            rayO, rayD = M + N * .0001, normalize(rayD - 2 * np.dot(rayD, N) * N)
            depth += 1
            # update the color
            col += reflection * col_ray
            reflection *= obj.get('reflection', 1.)
        img[h - j - 1, i, :] = np.clip(col, 0, 1)
plt.imsave('fig.png', img)