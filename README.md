# Supermassive-Black-Hole-Gas-and-Dark-Matter-Infall-Dynamics-code
We study black hole growth in a dwarf–galaxy. The dark-matter halo is described by an NFW distribution. We use N-particle simulation for dark matter and for Gas dynamics we use 1D Lagrangian shell scheme (spherical symmetry) solving the Euler equations for an ideal, adiabatic fluid. 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid as cumtrapz, quad
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import root_scalar
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# Constants
H0 = 67.4e3 / (3.086e22)  # Hubble constant in 1/s
sigma_T = 6.6524587e-29    # Thomson cross-section (m^2)
G = 6.67430e-11           # Gravitational constant
M_sun = 1.989e30          # Solar mass
c = 3e8                   # Speed of light
k_B = 1.38e-23            # Boltzmann constant
m_p = 1.6726e-27          # Proton mass
mu = 1.22  # Mean molecular weight - for neutral atomic gas
eta = 0.1                 # Radiative efficiency
PC = 3.086e16             # Parsec in meters
KPC = 1000 * PC
gamma = 5/3 # monoatomic ideal gas
T_Today_K= 2.7 # k, for bondy accretion rate in today time

# Cosmological Parameters 
Omega_m = 0.315
Omega_b = 0.049
Omega_dm = Omega_m - Omega_b
Omega_r = 9e-5
Omega_L = 1.0 - Omega_m - Omega_r
year = 365.25 * 24 * 3600  # Year in seconds
Myr = year * 1e6  # Megayear in seconds

# Inputs
#R_collapse = 0.8 * KPC  # Initial radius 
#R_collapse_same_density = 10 * KPC  # radius 
M_total = 1e9 * M_sun
#M_total = 1e9 * M_sun * (R_collapse / R_collapse_same_density)**3 # Total Mass of gas in kg
#M_total = 6.19e+36  # Total Mass of gas in kg for 1 radius mass 
#r_in_innermost = 0.3 * KPC   # innermost radius in meters
#r_out = R_collapse   # outermost radius in meters
angular_velocity_initial = 1e-14  # Initial angular velocity in rad/s 
angular_velocity_initial_DM = 1e-16  # Initial angular velocity in rad/s for DM 
z_collapse = 20 # Collapse redshift at which we start simulation, z time start from dark ages
z_collapse_end = 0 # Collapse redshift at which we end simulation 
initial_M_SMBH = 1e4 * M_sun # Initial black hole seed mass
concentration = 10 # NFW Dark Matter Profile Parameters tale how concentrated the dark matter is in inner region height mining higher accretion of dark matter currenly set to flatter
Virial_radius_multiplier = 10 # Virial radius multiplier for NFW profile, this is how much we want to scale the virial radius from initial radius


# computational input
N_shells = 100 # Number of radial shells for dark matter profile and gas profile
N_shells_evolution = 100 # Number of radial shells for gas profile evolution
N_shells_for_CDM = 1000 # Number of radial shells for CDM profile
num_steps = 1000 # Number of time steps in the while simulation
n_time_points = 2000 # Number of time steps for Gas shell simulation

# --- Time-Redshift Conversion Functions ---
# just change time to redshift or vice versa   
def E(z):
    return np.sqrt(Omega_r * (1 + z)**4 + Omega_m * (1 + z)**3 + Omega_L)

def Hubble(z):
    return H0 * E(z)

def lookback_integrand(z):
    return 1 / ((1 + z) * Hubble(z))

def cosmic_time(z):
    # Time from Big Bang to redshift z
    t_sec, _ = quad(lookback_integrand, z, np.inf)
    return t_sec

def z_of_t(t_sec):
#Convert cosmic time (seconds from Big Bang) to redshift
    # Find z where cosmic_time(z) = t_sec
    result = root_scalar(
        lambda z: cosmic_time(z) - t_sec,
        bracket=[0, 1000],
        method="brentq")
    return result.root if result.converged else None


#Convert shell model time to redshift for secondary axis
def time_to_redshift_for_plotting(time_array):
#Convert time array to redshift array for plotting
    z_array = np.zeros_like(time_array)
    for i, t in enumerate(time_array):
        z_array[i] = z_of_t(t)
    return z_array


# Setup Simulation Parameters 
t_start = cosmic_time(z_collapse)  # time at starting redshift
t_end = cosmic_time(z_collapse_end)            # present time
dt = (t_end - t_start) / num_steps  # timestep

# Time conversion to more readable units
years_per_second = 1 / (365.25 * 24 * 3600)
Myr_per_second = years_per_second * 1e-6

# Cloud properties
M_gas_tot_kg = M_total * (Omega_b / Omega_m)
M_dm_tot_kg = M_total * (Omega_dm / Omega_m)
rho_crit0 = 3 * H0**2 / (8 * np.pi * G)  # Critical density today
rho_b_z = rho_crit0 * Omega_b * (1 + z_collapse)**3
R_collapse = ((3 * M_gas_tot_kg) / (4 * np.pi * rho_b_z))**(1/3)  # cloud radius (m)

# Virial Temperature estimate
T_initial_K = (G * M_total * m_p) / (3 * k_B * R_collapse) # Accretion Power in Astrophysics" (Chapter 1, Eq 1.11) by Frank, King, and Raine

# Virial Temperature estimate
#T_initial_K = (G * M_total * m_p) / (3 * k_B * R_collapse_same_density) # just for same density but different radius
#T_initial_K =17380 # just for same density but different radius

# Accretion Rate Functions 
def eddington_accretion_rate(M_SMBH, eta):
#Calculate Eddington accretion rate in kg/s
    L_edd = 4 * np.pi * G * M_SMBH * m_p * c / sigma_T  # Eddington luminosity (W)
    return L_edd / (eta * c**2)  # Eddington accretion rate (kg/s)

def sound_speed(temperature):
#Calculate sound speed in m/s
    return np.sqrt(k_B * temperature / (mu * m_p))

def bondi_accretion_rate(M_SMBH, rho, c_s):
#Calculate Bondi accretion rate in kg/s
    return 4 * np.pi * G**2 * M_SMBH**2 * rho / (c_s**3)

def temperature_of_gas(temperature_of_interstellar_gas,z,gamma):
#Calculate temperature of gas at redshift z
    temperature_of_gas= temperature_of_interstellar_gas/((1+z)**( 3 * (gamma-1)))
    return temperature_of_gas


# --- NFW Profile Setup ---
#R_physical = R_collapse_same_density  * Virial_radius_multiplier 
#R_physical = R_collapse_same_density  * Virial_radius_multiplier
R_physical = R_collapse * Virial_radius_multiplier 
R_physical = R_collapse * Virial_radius_multiplier
r_s_nfw = R_physical / concentration
r_schwarzschild = 2 * G * initial_M_SMBH / c**2
r_min = max(r_schwarzschild, r_s_nfw * 0.01)
r_max = R_physical
r_grid = np.logspace(np.log10(r_min), np.log10(r_max), N_shells_for_CDM + 1)
r_centers = 0.5 * (r_grid[1:] + r_grid[:-1])

def rho_nfw(r, rho_0, r_s):
    x = r / r_s
    return rho_0 / (x * (1 + x)**2)

nfw_norm_factor = np.log(1 + concentration) - concentration / (1 + concentration)
rho_0_nfw = M_dm_tot_kg / (4 * np.pi * r_s_nfw**3 * nfw_norm_factor)
shell_volumes = 4/3 * np.pi * (r_grid[1:]**3 - r_grid[:-1]**3)
mass_shells_nfw = rho_nfw(r_centers, rho_0_nfw, r_s_nfw) * shell_volumes
mass_shells_nfw *= M_dm_tot_kg / np.sum(mass_shells_nfw)  # Normalize


# --- Run Simulations ---
def run_simulation(dark_matter_type="none"):
#Run simulation with specified dark matter type
    print(f"\nRunning simulation with {dark_matter_type} dark matter...")
    
    # Initialize
    M_SMBH = initial_M_SMBH
    M_gas_avail = M_gas_tot_kg

    # Initialize DM arrays inside the function so each run is independent
    dm_masses = mass_shells_nfw.copy()
    dm_radii = r_centers.copy()
    dm_velocities = np.zeros_like(dm_masses)
    dm_angular_momentum = dm_radii**2 * angular_velocity_initial_DM


    # Tracking arrays
    mass_history = [M_SMBH]
    gas_contrib_history = [0]
    dm_contrib_history = [0]
    time_history = [t_start]
    redshift_history = [z_collapse] 
    gas_masses = [M_gas_avail]
    dm_masses_total = [np.sum(dm_masses)]
    

    # Run simulation
    time = t_start
    step = 0
    while time < t_end:
        z_now = z_of_t(time)
        
        # For "CDM + Gas shell model", ignore Bondi gas accretion and use shell model BH mass
        if dark_matter_type == "CDM + Gas shell model":
            # Get the shell model BH mass at current time
            current_shell_mass = float(bh_mass_shell_interp(time))
            # Gas accretion comes from shell model (difference from previous mass)
            gas_accretion = current_shell_mass - M_SMBH
            gas_accretion = max(0, gas_accretion)  # Ensure non-negative
            #edd_rate = eddington_accretion_rate(M_SMBH, eta) * dt
            #gas_accretion = min(gas_accretion, edd_rate, M_gas_avail)
            # Update SMBH mass with shell model mass
            M_SMBH = current_shell_mass
            
            # Initialize DM accretion
            dm_accretion = 0
            
            # Dark matter accretion (same as other cases)
            if np.sum(dm_masses) > 0:
                accretion_radius = 3/2 * r_schwarzschild
                accreted_shells = 0
                for i in range(len(dm_masses)):
                    if dm_masses[i] <= 0:
                        continue
                    r = dm_radii[i]
                    L = dm_angular_momentum[i]
                    M_enclosed = np.sum(dm_masses[:i+1]) + M_SMBH
                    F_grav = -G * M_enclosed / r**2
                    F_cent = L**2 / r**3
                    a_radial = F_grav + F_cent
                    dm_velocities[i] += a_radial * dt
                    dm_radii[i] += dm_velocities[i] * dt
                    if dm_radii[i] < accretion_radius and dm_velocities[i] < 0:
                        M_SMBH += dm_masses[i]
                        dm_accretion += dm_masses[i]
                        dm_masses[i] = 0
                        dm_velocities[i] = 0
                        accreted_shells += 1
        
        else:
            # Gas accretion
            gas_accretion = 0
            dm_accretion = 0

            if M_gas_avail > 0:
                rho_gas = M_gas_avail / (4/3 * np.pi * R_physical**3)
                temperature_of_gas_with_time = temperature_of_gas(T_Today_K, z_now, gamma)
                c_s = sound_speed(temperature_of_gas_with_time)
                bondi_gas = bondi_accretion_rate(M_SMBH, rho_gas, c_s) * dt
                edd_rate = eddington_accretion_rate(M_SMBH, eta) * dt
                gas_accretion = min(bondi_gas, edd_rate, M_gas_avail)
                M_gas_avail -= gas_accretion

            # Dark matter accretion
            if dark_matter_type != "none" and np.sum(dm_masses) > 0:
                accretion_radius = 3/2 * r_schwarzschild
                accreted_shells = 0
                for i in range(len(dm_masses)):
                    if dm_masses[i] <= 0:
                        continue
                    r = dm_radii[i]
                    L = dm_angular_momentum[i]
                    M_enclosed = np.sum(dm_masses[:i+1]) + M_SMBH
                    F_grav = -G * M_enclosed / r**2
                    F_cent = L**2 / r**3
                    a_radial = F_grav + F_cent
                    dm_velocities[i] += a_radial * dt
                    dm_radii[i] += dm_velocities[i] * dt
                    if dm_radii[i] < accretion_radius and dm_velocities[i] < 0:
                        M_SMBH += dm_masses[i]
                        dm_accretion += dm_masses[i]
                        dm_masses[i] = 0
                        dm_velocities[i] = 0
                        accreted_shells += 1
            
            # Update black hole mass for non-shell models
            total_accretion = gas_accretion + dm_accretion
            M_SMBH += total_accretion

        # Update tracking arrays
        mass_history.append(M_SMBH)
        gas_contrib_history.append(gas_contrib_history[-1] + gas_accretion)
        dm_contrib_history.append(dm_contrib_history[-1] + dm_accretion)
        time_history.append(time)
        redshift_history.append(z_now)
        gas_masses.append(M_gas_avail)
        dm_masses_total.append(np.sum(dm_masses))

        # Advance time
        time += dt
        step += 1

        # Progress reporting
        if step % max(1, (num_steps // 10)) == 0:
            progress = (time - t_start) / (t_end - t_start) * 100
            print(f"Progress: {progress:.1f}%, z = {z_now:.2f}, BH Mass = {M_SMBH/M_sun:.2e} Msun")

    return {'mass_history': np.array(mass_history),
        'gas_contrib': np.array(gas_contrib_history),'dm_contrib': np.array(dm_contrib_history),
        'time_history': np.array(time_history),'redshift_history': np.array(redshift_history),
        'gas_masses': np.array(gas_masses),'dm_masses': np.array(dm_masses_total),}

#---------------------------------------------------------------------------------------------------------------------------------------

# Second simulation
# This code use same inoput but it attechd spperetly for gas accretion simulation privius one is bondi accretion
# it is based on lagrangion finite difference method, fore future finite volume method would be better then this
# Derived Parameters 
rho_gas_initial = M_gas_tot_kg / (4./3. * np.pi * R_collapse**3)  # Avg initial density
P_initial = rho_gas_initial * k_B * T_initial_K / (mu * m_p)  # Initial pressure
cs_initial = np.sqrt(k_B * T_initial_K / (mu * m_p))  # Initial sound speed

# NFW Dark Matter Profile Parameters
R_physical = R_collapse * Virial_radius_multiplier / (1 + z_collapse)
r_s_nfw = R_physical / concentration
# Central density parameter
rho_0 = M_dm_tot_kg / (4 * np.pi * r_s_nfw**3 * (np.log(1+concentration) - concentration/(1+concentration)))

# Enclosed mass from DM we use this in gas simulation code
def M_dm_enclosed(r):
    # NFW enclosed mass function
    x = r/r_s_nfw
    # Safe calculation for small x
    if isinstance(x, np.ndarray):
        result = np.zeros_like(x)
        Big_redius = x > 1e-10
        result[Big_redius] = 4 * np.pi * rho_0 * r_s_nfw**3 * (np.log(1 + x[Big_redius]) - x[Big_redius]/(1+x[Big_redius]))
        small_radius = x <= 1e-10
        result[small_radius] = 4 * np.pi * rho_0 * r_s_nfw**3 * (x[small_radius]**2/2)  # approximation for small x
        return result
    else:
        if x > 1e-10:
            return 4 * np.pi * rho_0 * r_s_nfw**3 * (np.log(1 + x) - x/(1+x))
        else:
            return 4 * np.pi * rho_0 * r_s_nfw**3 * (x**2/2)  # approximation for small x

# --- Setup for shell model ---
# Create shells with logarithmic spacing (more resolution near center)
shell_radii_fractions = np.logspace(-1.5, 0, N_shells+1)  # N_shells+1 boundaries give N_shells shells
# Logarithmic spacing between r_in and r_out
#shell_radii_fractions = np.logspace(np.log10(r_in_innermost) , np.log10(r_out) , N_shells+1)
shell_radii_fractions /= shell_radii_fractions[-1]  # Normalize to 1
initial_shell_boundaries = R_collapse * shell_radii_fractions

# just for same density but different radia
shell_boundaries = R_collapse * shell_radii_fractions

# Calculate initial shell properties
# jut putting all initial shell properties in arrays like mass,velocity,pressure,temperature, bla bla bla bla bla bla....

initial_shell_radii = 0.5 * (initial_shell_boundaries[1:] + initial_shell_boundaries[:-1])  # Shell midpoint
initial_shell_volumes = (4/3) * np.pi * (initial_shell_boundaries[1:]**3 - initial_shell_boundaries[:-1]**3)
#initial_shell_masses = M_gas_tot_kg * initial_shell_volumes / np.sum(initial_shell_volumes)   # Equal mass shells
initial_shell_masses = np.ones(N_shells) * M_gas_tot_kg / N_shells  # Equal mass shells
initial_shell_densities = initial_shell_masses / initial_shell_volumes

##shell_radii = 0.5 * (shell_boundaries[1:] + shell_boundaries[:-1])  # Shell midpoint # just for same density but different radia
#shell_volumes = (4/3) * np.pi * (initial_shell_boundaries[1:]**3 - initial_shell_boundaries[:-1]**3) # just for same density but different radia
##initial_shell_masses = M_gas_tot_kg * initial_shell_volumes / np.sum(initial_shell_volumes) 
##initial_shell_densities = initial_shell_masses / initial_shell_volumes # just for same density but different radia

initial_shell_pressures = initial_shell_densities * k_B * T_initial_K / (mu * m_p)
initial_shell_velocities = np.zeros(N_shells)  # Start from rest
initial_shell_temperatures = np.ones(N_shells) * T_initial_K
initial_specific_angular_momentum = initial_shell_radii**2 * angular_velocity_initial

# Calculate initial entropy constant for each shell (K in P = K * rho^gamma)
initial_entropy_constants = initial_shell_pressures / initial_shell_densities**gamma

print(f"Model initialized with {N_shells} shells")
print(f"Initial cloud radius: {R_collapse/PC/1000:.2f} kpc")
print(f"Initial gas density: {rho_gas_initial:.2e} kg/m³")
print(f"Initial temperature: {T_initial_K:.2f} K")
print(f"Initial sound speed: {cs_initial:.2e} m/s")
print(f"Dark matter scale radius: {r_s_nfw/PC/1000:.2f} kpc")
print(f"Innermost shell innermost boundary radius initial radius: {initial_shell_boundaries[1]/(PC*1000):.4f} kpc")
print(f"Initial shell initial mass: {initial_shell_masses[0]:.2e} Kg")
print(f"Outermost shell initial radius: {initial_shell_boundaries[-1]/(PC*1000):.4f} kpc")

# --- Define the ODE function ---
# This we doing for Continuity equation for density ρ and radial velocity v
# Just store all state variables in one vector like radius, velocities, densities that we accessing from this
# State vector: [r_1, r_2, ..., r_N, v_1, v_2, ..., v_N, rho_1, rho_2, ..., rho_N]
def radial_collapse(t,y):
    # Extract components from state vector
    radii = y[:N_shells]  # Current shell radii
    velocities = y[N_shells:2*N_shells]  # Current shell velocities
    densities = y[2*N_shells:3*N_shells]  # Current shell densities
    
    # Output vectors for derivatives
    drdt = velocities  # dr/dt = velocity
    dvdt = np.zeros(N_shells)  # Initialize dv/dt
    drhodt = np.zeros(N_shells)  # Initialize drho/dt
    
    # Safety check for very small radii
    radii = np.maximum(radii, 1e10)  # Minimum radius (~0.001 light-year)
    
    # Sort shells by radius (in case of shell crossing)
    sort_indices = np.argsort(radii)
    masses_sorted = initial_shell_masses[sort_indices]
    
    # Calculate enclosed mass for each sorted radius
    M_gas_enclosed = np.cumsum(masses_sorted)
    
    # Calculate new pressures from current densities using adiabatic EoS
    pressures = initial_entropy_constants * densities**gamma
    
    # Calculate accelerations for each shell
    for i in range(N_shells_evolution):
        r_i = radii[i]
        rho_i = densities[i]
        v_i = velocities[i]
        L_per_m = initial_specific_angular_momentum[i]  # Conserved specific angular momentum

        # Find this shell's position in sorted array
        shell_is_sorted = np.where(sort_indices == i)[0][0]

        # Calculate enclosed gas mass
        if shell_is_sorted > 0:
            M_gas_enc = M_gas_enclosed[shell_is_sorted-1]
        else:
            M_gas_enc = 0.0

        # Add half of this shell's mass (for self-gravity)
        M_gas_enc += initial_shell_masses[i] / 2

        # Calculate enclosed dark matter mass from NFW profile
        M_dm_enc = M_dm_enclosed(r_i)

        # --- Force calculations ---

        # 1. Gravitational force (gas + dark matter + black hole)
        F_gravity = -G * (M_gas_enc + M_dm_enc + initial_M_SMBH) / r_i**2

        # 2. Pressure gradient force
        # first we do for inner most shell and then for outermost shell and then for all other shells if do not do this code going to confused on boundry cunditions
        # Estimate pressure gradient using neighboring shells
        if i == 0:  # Innermost shell
            # Use one-sided difference
            dr = radii[1] - radii[0]
            dP = pressures[1] - pressures[0]
            # For stability at center, use stronger central pressure
            P_gradient = dP / dr if dr > 0 else 0
        elif i == N_shells - 1:  # Outermost shell
            # Use one-sided difference
            dr = radii[i] - radii[i-1]
            dP = pressures[i] - pressures[i-1]
            P_gradient = dP / dr if dr > 0 else 0
        else:  # Interior shells
            # Use central difference
            dr = radii[i+1] - radii[i-1]
            dP = pressures[i+1] - pressures[i-1]
            P_gradient = dP / dr if dr > 0 else 0

        F_pressure = -P_gradient / rho_i  # Force per unit mass

        # 3. Centrifugal force
        F_centrifugal = L_per_m**2 / r_i**3

        # Total acceleration
        dvdt[i] = F_gravity + F_pressure + F_centrifugal

        # 4. Density evolution (from continuity equation)
        # same as we done for pressure gradient we do for density evolution
        if i == 0:  # Innermost shell
            dv_dr = (velocities[1] - velocities[0]) / (radii[1] - radii[0]) if radii[1] > radii[0] else 0
        elif i == N_shells - 1:  # Outermost shell
            dv_dr = (velocities[i] - velocities[i-1]) / (radii[i] - radii[i-1]) if radii[i] > radii[i-1] else 0
        else:  # Interior shells
            dv_dr = (velocities[i+1] - velocities[i-1]) / (radii[i+1] - radii[i-1]) if radii[i+1] > radii[i-1] else 0
        # this just a Continuity equation for density ρ and radial velocity v
        # ∂ρ/∂t + (1/r^2) ∂/∂r (r^2 ρ v) = 0 become ∂ρ/∂t = -ρ * (2v/r + ∂v/∂r)
        drhodt[i] = -rho_i * (2*v_i/r_i + dv_dr)
    
    # Combine all derivatives into a single state vector
    return np.concatenate((drdt, dvdt, drhodt))

# --- Initial state vector ---
# Initial state vector: [r_1, r_2, ..., r_N, v_1, v_2, ..., v_N, rho_1, rho_2, ..., rho_N]
y0 = np.concatenate((initial_shell_radii,initial_shell_velocities, initial_shell_densities))

# Create logarithmically spaced time points from t_start to t_end
t_eval = np.geomspace(t_start, t_end, n_time_points)

# Add additional times at the beginning for better resolution of early dynamics
# Make sure these are all >= t_start
t_early_duration = min(1000 * year, (t_end - t_start) * 0.1)  # First 1000 years or 10% of total time
t_early = np.geomspace(t_start, t_start + t_early_duration, 20)[:-1]  
t_eval = np.concatenate((t_early, t_eval))
t_eval = np.unique(np.sort(t_eval))  # Remove duplicates and sort

print(f"Integration time range: {t_start/(1e9*year):.3f} to {t_end/(1e9*year):.3f} Gyr")
print("Starting integration...")

# I used first rk4 but simulation just blasted so I use LSODA, LSODA is come from scipy.integrate because we cannot use cupy from cuda scipy dose not supported by cuda
# for any one in future reading this code and whant to do batter gas simulation have to use 3d finite volume method with c++ leguage and cuda for GPU acceleration ,c++ bater supported by cuda

#solve_ivp: this come from scipy integretion library
#radial_collapse: function that computes the derivatives (how radii, velocities, and densities change).
#[t_start, t_end]: The time range for the simulation (from start to end).
#y0: The initial state vector (initial radii, velocities, densities for all shells).
#t_eval: The specific time points where you want the solution evaluated.
#method='LSODA': Uses the LSODA solver, which automatically switches between stiff and non-stiff methods.
#rtol=1e-6, atol=1e-9: Sets the error tolerances for the integration.

# LSODA automatically switches between non-stiff and stiff methods as needed
# Non-stiff: Standard ODEs, can be solved with explicit methods.
# Stiff: ODEs where some components change much faster than others, requiring implicit methods for stability. 
# just to control error if error go over this bundry the LSODA swiches to stiff method
# and if error is low it switches to non-stiff method
# this is why rk4 dose not work because we have stiff ODEs in this simulation

#rtol (relative tolerance): Controls the allowed relative error in each integration step.
#Example: 1e-6 means the error can be up to one part in a million of the current value.
#atol (absolute tolerance): Controls the allowed absolute error in each integration step.
#Example: 1e-9 means the error can be up to 1e-9 in absolute value.

sol = solve_ivp(radial_collapse, [t_start, t_end], y0, t_eval=t_eval, method='LSODA',rtol=1e-6,atol=1e-9)
print("Integration complete!")

# Just for checking if integration was successful this line is not necessary but it is good to have
if not sol.success:
    print("WARNING: Integration did not complete successfully!")
    print("Message:", sol.message)

# --- Process results ---

# Extract results
# just extrecting all results that we put from integretion 
time_Gyr = np.array(sol.t) / (1e9 * year)  # Convert seconds to Gyr
time_Myr = np.array(sol.t) / Myr  # Also keep Myr for short times
radii_kpc = sol.y[:N_shells] / (PC * 1000)  # Convert m to kpc
velocities_km_s = sol.y[N_shells:2*N_shells] / 1000  # Convert m/s to km/s
densities = sol.y[2*N_shells:3*N_shells]  # kg/m³

# Calculate temperatures using adiabatic relation
temperatures = np.zeros((N_shells, len(sol.t)))
for i in range(N_shells):
    for j in range(len(sol.t)):
        # P = K * rho^gamma and P = rho * k_B * T / (mu * m_p)
        # So T = K * rho^(gamma-1) * (mu * m_p) / k_B
        temperatures[i, j] = initial_entropy_constants[i] * densities[i, j]**(gamma-1) * (mu * m_p) / k_B


# v_rot = L / r, where L is the specific angular momentum (constant)
v_rot_m_s = initial_specific_angular_momentum[0] / (sol.y[0, :])  # m/s
v_rot_km_s = v_rot_m_s / 1000

# --- Innermost shell temperature, pressure, and density ---
# we only nees inner most shell for accretion rate and black hole mass growth
innermost_density = densities[0, :]
innermost_temperature = temperatures[0, :]
innermost_pressure = initial_entropy_constants[0] * innermost_density**gamma

# --- Accretion rate onto black hole (innermost shell) ---
# This is a standard thin disk accretion formula, combining local disk properties (density, temperature, rotation, inflow) to estimate how much mass flows inward per unit time.
# we need because if we shift inner most shell at schwarzschild radius the gas code dose not work so we need the formula we feed in this the inner most shell parameter from gas simulation
# Accretion Power in Astrophysics" (Chapter 5) by Frank, King, and Raine
# The formula is Mdot = 2 * pi * r * Sigma * v_r, where Sigma is the surface density and v_r is the radial velocity.
# We assume that the gas behaves like a thin disk with a scale height H determined by the sound speed and rotation frequency.
def accretion_rate_radial(r, rho, T, v_phi, v_r):
    cs = np.sqrt(k_B * T / (mu * m_p))
    Omega = v_phi / r
    # Avoid division by zero element-wise
    H = np.where(Omega != 0, cs / Omega, 1e-10)
    Sigma = rho * H
    Mdot = 2 * np.pi * r * Sigma * v_r
    return Mdot

# Calculate v_phi (rotational velocity) and v_r (radial velocity) for innermost shell
v_phi = v_rot_m_s  # already computed for innermost shell, shape (nt,)
v_r = sol.y[N_shells, :]  # radial velocity for innermost shell, shape (nt,)
r_in = sol.y[0, :]  # radius for innermost shell, shape (nt,)
rho_in = innermost_density
T_in = innermost_temperature

accretion_rate = accretion_rate_radial(r_in, rho_in, T_in, v_phi, v_r)  

# --- Black hole mass growth (integrated accretion) ---
# Set negative accretion rates to zero (no mass loss)
accretion_rate_pos = np.where(accretion_rate > 0, accretion_rate, 0)
# Integrate using cumulative trapezoid rule
bh_mass_growth = initial_M_SMBH + cumulative_trapezoid(accretion_rate_pos, sol.t, initial=0)
bh_mass_growth_Msun = bh_mass_growth / M_sun

# Plot innermost shell rotational (tangential) velocity vs time
# v_rot = L / r, where L is the specific angular momentum (constant)
v_rot_m_s = initial_specific_angular_momentum[0] / (sol.y[0, :])  # m/s
v_rot_km_s = v_rot_m_s / 1000 #

# this in while loop with CDM model mass update
# Interpolate BH mass from shell model as a function of time
bh_mass_shell_interp = interp1d(sol.t, bh_mass_growth, bounds_error=False, fill_value=(bh_mass_growth[0], bh_mass_growth[-1]))

# --- Run three simulations ---
#results_no_dm = run_simulation("none")
#results_cdm = run_simulation("CDM")
results_cdm_gas_shell_model = run_simulation("CDM + Gas shell model")

# Convert time to Myr for plotting
#time_Myr_no_dm = results_no_dm['time_history'] * Myr_per_second
#time_Myr_cdm = results_cdm['time_history'] * Myr_per_second
time_Myr_cdm_gas_shell = results_cdm_gas_shell_model['time_history'] * Myr_per_second

#Create redshift arrays for all time series
shell_redshift = time_to_redshift_for_plotting(sol.t)


# --- Plot Results ---
# Convert shell model time to Myr for plotting
shell_time_Myr = time_Gyr * 1000  # Convert Gyr to Myr
fig, axes = plt.subplots(1, 2, figsize=(16, 12))

# Plot 1: Black hole mass growth comparison
ax1 = axes[0]
#ax1.plot(time_Myr_no_dm, results_no_dm['mass_history'] / M_sun, 'b-', label='No Dark Matter')
#ax1.plot(time_Myr_cdm, results_cdm['mass_history'] / M_sun, 'r-', label='CDM (NFW)')
ax1.plot(time_Myr_cdm_gas_shell, results_cdm_gas_shell_model['mass_history'] / M_sun, 'g-', linewidth=2, label='CDM + Gas Shell Model')
ax1.plot(shell_time_Myr, bh_mass_growth_Msun, 'k--', linewidth=2, label='Gas Only (Shell Model)')
ax1.set_title('Black Hole Mass Growth')
ax1.set_xlabel('Time (Myr since Big Bang)')
ax1.set_ylabel('Black Hole Mass (Solar Masses)')
ax1.set_yscale('log')
#ax1.set_xscale('log')
ax1.legend()
ax1.grid(True, which="both", ls="-", alpha=0.2)
ax1_top = ax1.twiny()
# Select a subset of time points for redshift labels to avoid crowding
time_subset_indices = np.linspace(0, len(shell_time_Myr)-1, 8, dtype=int)
time_subset = shell_time_Myr[time_subset_indices]
z_subset = shell_redshift[time_subset_indices]
ax1_top.set_xlim(ax1.get_xlim())
ax1_top.set_xticks(time_subset)
ax1_top.set_xticklabels([f'{z:.1f}' for z in z_subset])
ax1_top.set_xlabel('Redshift (z)', fontsize=12)
#ax1_top.set_xscale('log')


# Plot 2: Contribution breakdown
ax2 = axes[1]
#ax2.plot(time_Myr_cdm, results_cdm['gas_contrib'] / M_sun, 'b--', label='Gas Contribution (CDM)')
#ax2.plot(time_Myr_cdm, results_cdm['dm_contrib'] / M_sun, 'r--', label='DM Contribution (CDM)')
ax2.plot(time_Myr_cdm_gas_shell, results_cdm_gas_shell_model['gas_contrib'] / M_sun, 'g-', linewidth=2, label='Gas Contribution (CDM + Shell)')
ax2.plot(time_Myr_cdm_gas_shell, results_cdm_gas_shell_model['dm_contrib'] / M_sun, 'orange', linewidth=2, label='DM Contribution (CDM + Shell)')
shell_gas_contrib = bh_mass_growth_Msun - (initial_M_SMBH / M_sun)
ax2.plot(shell_time_Myr, shell_gas_contrib, 'k--', linewidth=2, label='Gas Contribution (Shell Model)')
ax2.set_title('Contribution to BH Growth')
ax2.set_xlabel('Time (Myr since Big Bang)')
ax2.set_ylabel('Mass Contribution (Solar Masses)')
ax2.set_yscale('log')
#ax1.set_xscale('log')
ax2.legend()
ax2.grid(True, which="both", ls="-", alpha=0.2)
ax2_top = ax2.twiny()
ax2_top.set_xlim(ax2.get_xlim())
ax2_top.set_xticks(time_subset)
ax2_top.set_xticklabels([f'{z:.1f}' for z in z_subset])
ax2_top.set_xlabel('Redshift (z)', fontsize=12)
#ax1_top.set_xscale('log')

# Add text annotation showing final DM contribution
#final_dm_contrib = results_cdm['dm_contrib'][-1] / M_sun
#final_gas_contrib = results_cdm['gas_contrib'][-1] / M_sun
final_dm_contrib_shell = results_cdm_gas_shell_model['dm_contrib'][-1] / M_sun
final_gas_contrib_shell = results_cdm_gas_shell_model['gas_contrib'][-1] / M_sun
#ax2.text(0.05, 0.95, f'CDM Model:\nDM: {final_dm_contrib:.2e} M☉\nGas: {final_gas_contrib:.2e} M☉\n\nCDM + Shell Model:\nDM: {final_dm_contrib_shell:.2e} M☉\nGas: {final_gas_contrib_shell:.2e} M☉', 
         #transform=ax2.transAxes, verticalalignment='top', 
         #bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax2.text(0.05, 0.95, f'CDM + Shell Model:\nDM: {final_dm_contrib_shell:.2e} M☉\nGas: {final_gas_contrib_shell:.2e} M☉', 
         transform=ax2.transAxes, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()

# --- Print Black Hole Growth Summary ---
print("\n" + "="*60)
print("BLACK HOLE GROWTH SUMMARY")
print("="*60)
print(f"Initial BH Mass: {initial_M_SMBH/M_sun:.2e} M☉")
print()

# No Dark Matter simulation
#final_mass_no_dm = results_no_dm['mass_history'][-1]
#growth_factor_no_dm = final_mass_no_dm / initial_M_SMBH
#print(f"No Dark Matter:")
#print(f"  Final Mass: {final_mass_no_dm/M_sun:.2e} M☉")
#print(f"  Growth Factor: {growth_factor_no_dm:.1f}×")
#print(f"  Mass Added: {(final_mass_no_dm - initial_M_SMBH)/M_sun:.2e} M☉")
#print()

# CDM simulation
#final_mass_cdm = results_cdm['mass_history'][-1]
#growth_factor_cdm = final_mass_cdm / initial_M_SMBH
#total_mass_added = final_mass_cdm - initial_M_SMBH
#print(f"CDM (NFW Profile):")
#print(f"  Final Mass: {final_mass_cdm/M_sun:.2e} M☉")
#print(f"  Growth Factor: {growth_factor_cdm:.1f}×")
#print(f"  Mass Added: {total_mass_added/M_sun:.2e} M☉")
#print(f"  Gas Contribution: {results_cdm['gas_contrib'][-1]/M_sun:.2e} M☉ ({results_cdm['gas_contrib'][-1]/total_mass_added*100:.1f}%)")
#print(f"  DM Contribution: {results_cdm['dm_contrib'][-1]/M_sun:.2e} M☉ ({results_cdm['dm_contrib'][-1]/total_mass_added*100:.1f}%)")
#print(f"  Total DM available initially: {M_dm_tot_kg/M_sun:.2e} M☉")
#print(f"  DM accreted fraction: {results_cdm['dm_contrib'][-1]/M_dm_tot_kg*100:.3f}%")
#print()

# CDM + Gas Shell Model simulation
final_mass_cdm_shell = results_cdm_gas_shell_model['mass_history'][-1]
growth_factor_cdm_shell = final_mass_cdm_shell / initial_M_SMBH
total_mass_added_shell = final_mass_cdm_shell - initial_M_SMBH
print(f"CDM + Gas Shell Model:")
print(f"  Final Mass: {final_mass_cdm_shell/M_sun:.2e} M☉")
print(f"  Growth Factor: {growth_factor_cdm_shell:.1f}×")
print(f"  Mass Added: {total_mass_added_shell/M_sun:.2e} M☉")
print(f"  Gas Contribution: {results_cdm_gas_shell_model['gas_contrib'][-1]/M_sun:.2e} M☉ ({results_cdm_gas_shell_model['gas_contrib'][-1]/total_mass_added_shell*100:.1f}%)")
print(f"  DM Contribution: {results_cdm_gas_shell_model['dm_contrib'][-1]/M_sun:.2e} M☉ ({results_cdm_gas_shell_model['dm_contrib'][-1]/total_mass_added_shell*100:.1f}%)")
print(f"  Total DM available initially: {M_dm_tot_kg/M_sun:.2e} M☉")
print(f"  DM accreted fraction: {results_cdm_gas_shell_model['dm_contrib'][-1]/M_dm_tot_kg*100:.3f}%")
print()

# Shell Model simulation
final_mass_shell = bh_mass_growth[-1]
growth_factor_shell = final_mass_shell / initial_M_SMBH
total_gas_accreted = final_mass_shell - initial_M_SMBH
print(f"Gas Shell Model:")
print(f"  Final Mass: {final_mass_shell/M_sun:.2e} M☉")
print(f"  Growth Factor: {growth_factor_shell:.1f}×")
print(f"  Mass Added: {total_gas_accreted/M_sun:.2e} M☉")
print(f"  Gas Contribution: {total_gas_accreted/M_sun:.2e} M☉ (100.0%)")
print()
