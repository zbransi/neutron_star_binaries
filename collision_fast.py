from numba import njit
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from numpy.linalg import norm
from tqdm import tqdm

G = 6.67e-8
solar_mass = 1.989e+33
sec_per_day = 86400
solar_radius = 6.957e+10

def unit(vec):
    return vec / norm(vec)

@njit
def init_position(m1, m2, a, e):
    mu = m1 + m2
    factor = a * m1 / mu
    r1 = np.array([factor * (1.0 + e), 0.0, 0.0])
    r2 = np.array([- factor * (1.0 + e), 0.0, 0.0])
    return r1, r2

@njit
def init_velocity(m1, m2, r1, r2, a, e):
    dist = norm(r2 - r1)
    mu = m1 + m2
    if e == 0:
       force = G * m1 * m2 / (dist ** 2)
       v1 = np.array([0.0, np.sqrt(force * norm(r1) / m1), 0.0])
       v2 = np.array([0.0, - np.sqrt(force * norm(r2) / m2), 0.0])
       return v1, v2
    if dist == 0 or a == 0:
        raise ValueError('Positions of bodies coincide, leading to infinite velocity.')
    v_rel = np.sqrt(G * mu * ((2 / dist) - (1 / a)))
    v1 = np.array([0.0, m2 * v_rel / mu, 0.0])
    v2 = np.array([0.0, - m1 * v_rel / mu, 0.0])
    return v1, v2
    

@njit
def compute_gravity(m1, m2, r1, r2):
    dist_vec = r2 - r1
    dist_mag = np.linalg.norm(dist_vec)
    if dist_mag == 0:
        raise ValueError('Positions of bodies coincide, leading to infinite force.')
    unit_vec = dist_vec / dist_mag
    g1 = G * m2 / (dist_mag ** 2) * unit_vec
    g2 = -G * m1 / (dist_mag ** 2) * unit_vec     
    return g1, g2

@njit
def update_position(r, v, dt):
    return r + v * dt

@njit
def update_velocity_half(v, a, dt):
    return v + 0.5 * a * dt

@njit
def compute_momentum(m1, m2, v1, v2):
    p1 = m1 * v1
    p2 = m2 * v2
    return p1 + p2

@njit
def compute_energy(m1, m2, v1, v2, r1, r2):
    dist = norm(r1 - r2)
    total_gravitational = - G * m1 * m2 / dist
    kinetic1 = 0.5 * m1 * np.dot(v1, v1)
    kinetic2 = 0.5 * m2 * np.dot(v2, v2)
    total_kinetic = kinetic1 + kinetic2
    total_energy = total_gravitational + total_kinetic
    return total_energy

@njit
def random_kick(v_kick):
    x1, x2 = np.random.rand(2)
    theta = np.arccos(1 - (2 * x1))
    phi = 2 * np.pi * x2

    vx = v_kick * np.sin(theta) * np.cos(phi)
    vy = v_kick * np.sin(theta) * np.sin(phi)
    vz = v_kick * np.cos(theta)

    return np.array([vx, vy, vz]), theta, phi

# m1 is the Ns and m2 is the companion
@njit
def apply_kick(v_kick_vec, m1, m2, v1, v2):
    new_v1 = v1 + v_kick_vec
    new_v2 = - m1 * new_v1 / m2

    return new_v1, new_v2

@njit
def orbital_elements(m1, m2, v1, v2, r1, r2):
    mu = G * (m1 + m2)

    r_vec = r1 - r2
    v_vec = v1 - v2

    r_mag = norm(r_vec)
    v_mag = norm(v_vec)

    h_vec = np.cross(r_vec, v_vec)
    h = norm(h_vec)

    epsilon = (v_mag ** 2) / 2 - mu / r_mag
    if epsilon >= 1e-8:
        return 0, np.nan, np.nan, np.nan
    
    a = - mu / (2 * epsilon)
    e = np.sqrt(1 + (2 * epsilon * h**2) / (mu**2))
    T = 2 * np.pi * np.sqrt(a**3 / (G * (m1 + m2)))

    return 1, a, e, T

@njit
def step(m1, m2, v1, v2, r1, r2, g1, g2, dt):
    v1 = update_velocity_half(v1, g1, dt)
    v2 = update_velocity_half(v2, g2, dt)

    r1 = update_position(r1, v1, dt)
    r2 = update_position(r2, v2, dt)

    g1, g2 = compute_gravity(m1, m2, r1, r2)
    
    v1 = update_velocity_half(v1, g1, dt)
    v2 = update_velocity_half(v2, g2, dt)

    return r1, r2, v1, v2, g1, g2

def timing(module, period, dt):
    match module:
        case 'elliptical':
            return int((np.random.rand() / 2) * period / dt) 
        case 'periapsis':
            return int(0.5 * period / dt)
        case _:
            return 0

data_cols= ['period_init', 'v_kick', 'collide?', 'bound?']

def trial(module='circular', period=100, v_kick=50, e=0, m1=10, m2=10):
    T_init = period * sec_per_day
    a_init = (G * solar_mass * (m1 + m2) / (4 * (np.pi ** 2))) ** (1/3) * (T_init ** (2/3))
    e_init = e
    m1_init = m1 * solar_mass
    m2_init = m2 * solar_mass
    v_kick = v_kick * (10 ** 5)
    v_kick_vec, theta, phi = random_kick(v_kick)
    rad1, rad2 = 2.5 * solar_radius, 2.5 * solar_radius

    dt = T_init / 1000

    r1, r2 = init_position(m1_init, m2_init, a_init, e_init)
    v1, v2 = init_velocity(m1_init, m2_init, r1, r2, a_init, e_init)

    g1, g2 = compute_gravity(m1_init, m2_init, r1, r2)
    momentum = compute_momentum(m1_init, m2_init, v1, v2)
    energy = compute_energy(m1_init, m2_init, v1, v2, r1, r2)
    collide = False

    for _ in range(timing(module, T_init, dt)):
        r1, r2, v1, v2, g1, g2 = step(m1_init, m2_init, v1, v2, r1, r2, g1, g2, dt)

    m1_fin = 1.4 * solar_mass
    m2_fin = m2_init
    rad1 = 1.1e+6
    v1, v2 = apply_kick(v_kick_vec, m1_fin, m2_fin, v1, v2)
    momentum = compute_momentum(m1_fin, m2_fin, v1, v2)
    energy = compute_energy(m1_fin, m2_fin, v1, v2, r1, r2)
    bound, a_fin, e_fin, T_fin = orbital_elements(m1_fin, m2_fin, v1, v2, r1, r2)
    tighten = a_fin < a_init
    circularize = e_fin < e_init
    g1, g2 = compute_gravity(m1, m2, r1, r2)

    for _ in range(200):
        r1, r2, v1, v2, g1, g2 = step(m1_fin, m2_fin, v1, v2, r1, r2, g1, g2, dt)
        dist = norm(r2 - r1)
        if dist < (rad1 + rad2):
            collide = True
            break

    return np.array([T_init, v_kick, collide, bound])

def run_trial(args):
    module, period, v_kick = args
    e = 0 if module == 'circular' else 0.5
    return trial(module=module, period=period, v_kick=v_kick, e=e)

MODULE = 'circular'

period_range = np.linspace(1, 1000, 100) # From 1 to 1000 days
kick_velocity_range = np.linspace(1, 500, 100) # From 1 to 500 km/s

chunk_size = 1000
summary_results = []

with ProcessPoolExecutor(max_workers=6) as executor:
    config_list = [(p, k) for p in period_range for k in kick_velocity_range]
    for i, (period, v_kick) in enumerate(tqdm(config_list, desc="Configs")):
        chunk = [(MODULE, period, v_kick) for _ in range(chunk_size)]
        futures = [executor.submit(run_trial, args) for args in chunk]

        chunk_results = []
        for future in tqdm(as_completed(futures), total=len(futures), leave=False, desc=f"Chunk {i+1}"):
            chunk_results.append(future.result())

        collide_values = [r[2] for r in chunk_results]
        bound_values = [r[3] for r in chunk_results]

        collide_prob = np.mean(collide_values)
        bound_prob = np.mean(bound_values)
        
        summary_results.append([period, v_kick, collide_prob, bound_prob])
    
modelData = pd.DataFrame(summary_results, columns=data_cols)
modelData.to_csv(f"data_resol/collide_{MODULE}.csv", index=False)