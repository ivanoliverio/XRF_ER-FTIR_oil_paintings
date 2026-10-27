#!/usr/bin/env python
# coding: utf-8

"""Physics-based XRF simulation and synthetic paint-stratigraphy generation.

Main workflow:
1. Sample pigment compositions, binder fractions, thicknesses, and ground layers.
2. Convert each stratigraphy into elemental compositions and XRF layer inputs.
3. Compute primary fluorescence, intra-/inter-layer enhancement, and attenuation.
4. Apply detector efficiency and Voigt line broadening to obtain spectra.
5. Serialize spectra and layer metadata for downstream neural-network training.

Physical calculations and numerical parameters are preserved from the source.
"""


# In[12]:

#Geometrical factors 
A             = 20.0 #Detector active area
d_geom             = 14.0 #Detector-to-sample distance
phi           = [1.18682, 0.907571] #incident and emission angles [rad[


# In[13]:

#utility imports -> requires xraydb
#!pip install xraydb

from xraydb import material_mu
import xraydb as xraydb
import numpy as np
import matplotlib.pyplot as plt
from xraydb import mu_elam
from xraydb import fluor_yield
from xraydb import ck_probability
from xraydb import atomic_number
import matplotlib.ticker as ticker
import scipy.integrate as integrate
from scipy.integrate import quad
from numpy import sin, cos, tan
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xraydb
import xraylib
import re
import collections

# In[7]:

plt.rcParams["figure.dpi"] = 120

# ============================================================
#  Parameters and File path
# ============================================================

PIGMENT_FILE   = "Pigment_range.json"       # file with pigment compositional ranges
DENSITY_FILE   = "Density.json"   # file with density of the pigment compounds
C_DIRICHLET    = 20.0                       # Dirichlet alpha factor
BETA_A         = 3.0                        # Parameters of the Beta distribution for the binder-to-pigment mass ratio
BETA_B         = 3.0
LAMBDA_LAYERS  = 1.5                        # Poisson mean for the number of layer
R_MIN          = 0.5                        # min binder-to-pigment mass ratio
R_MAX          = 4.0                        # max binder-to-pigment mass ratio
RHO_OIL        = 0.93                       # binder density [g/cm^3] 
RANDOM_SEED    = None                         

rng = np.random.default_rng(RANDOM_SEED)

# ============================================================
#  JSON files 
# ============================================================

with open(PIGMENT_FILE, "r") as f:
    pigment_data = json.load(f)

with open(DENSITY_FILE, "r") as f:
    density_data = json.load(f)

pigment_names = list(pigment_data.keys())
#print("Available Pigments", len(pigment_names))

# ============================================================
#  Utility:atomic masses from xraydb
# ============================================================

_atomic_mass_cache = {}

def atomic_mass(symbol: str) -> float:
    """Return the atomic mass (g/mol) of an element using xraydb."""
    symbol = symbol.strip()
    if symbol not in _atomic_mass_cache:
        _atomic_mass_cache[symbol] = xraydb.atomic_mass(symbol)
    return _atomic_mass_cache[symbol]

# ============================================================
#  Chemical formula parser
# ============================================================

def normalize_formula_string(formula: str) -> str | None:
    """Normalize chemical formulas and substitute supported approximations for complex or non-stoichiometric compounds. Return None for formulas that cannot be resolved."""
    if formula is None:
        return None

    formula = formula.strip()

    # strip trailing annotations such as "C16H10N2O2 (isomer)"
    if " " in formula and not "(" in formula.split(" ")[0]:
        first = formula.split(" ")[0]
        # use the first token if it looks like a valid chemical formula
        if re.match(r'^[A-Z][A-Za-z0-9().·+/]*$', first):
            formula = first

    # override for non-stoichiometric or mixed cases
    overrides = {
        "Fe2O3/FeO(OH)": "Fe2O3",
        "SiO2 / Al2Si2O5(OH)4": "SiO2",
        "SiO2 / CaCO3": "SiO2",
        "PbSnO3 / PbSn1−xSixO3": "PbSnO3",
        "PbSn1−xSixO3": "PbSnO3",
        "K(Mg,Fe2+)(Fe3+,Al)(Si4O10)(OH)2": "KFeMgAlSi4O10(OH)2",
        "(K,Na)(Fe3+,Al,Mg)2(Si4O10)(OH)2": "KFe2Si4O10(OH)2",
        "K0.7Al2(Al,Si)4O10(OH)2": "K0.7Al3Si3O10(OH)2",
        "Al3+ (complex)": "Al2O3",
        "C_xH_y": None,
        "C_xH_yO_z": None
    }

    if formula in overrides:
        return overrides[formula]

    
    formula = formula.replace("·", "+").replace("−", "-").replace(" ", "")

 
    if re.search(r'_[xyzXYZ]', formula):
        return None

    return formula


def parse_formula(formula: str) -> dict:
    """Parse a chemical formula into a mapping of element symbols to stoichiometric coefficients. Supports parentheses, decimal coefficients, and hydrate additions."""
    formula = normalize_formula_string(formula)
    if formula is None:
        raise ValueError("Generic formula / non parseable.")

    #handle hydrates and similar (e.g. CaSO4+2H2O)
    groups = formula.split("+")

    num_pattern = r'\d*\.?\d+'    
    token_pattern = rf'([A-Z][a-z]?|\(|\)|{num_pattern})'

    def parse_group(s: str) -> collections.Counter:
        tokens = re.findall(token_pattern, s)
        stack = [collections.Counter()]
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == '(':
                stack.append(collections.Counter())
                i += 1
            elif tok == ')':
                group = stack.pop()
                i += 1
                mult = 1.0
                if i < len(tokens) and re.fullmatch(num_pattern, tokens[i]):
                    mult = float(tokens[i])
                    i += 1
                for el, count in group.items():
                    stack[-1][el] += count * mult
            elif re.fullmatch(num_pattern, tok):
                
                raise ValueError(f"Isolated number in formula: {s}")
            else:
                # element
                el = tok
                i += 1
                mult = 1.0
                if i < len(tokens) and re.fullmatch(num_pattern, tokens[i]):
                    mult = float(tokens[i])
                    i += 1
                stack[-1][el] += mult

        if len(stack) != 1:
            raise ValueError("Parentesi non bilanciate in formula: " + s)
        return stack[0]

    total = collections.Counter()
    for g in groups:
        g = g.strip()
        if not g:
            continue
        m = re.match(rf'^({num_pattern})(.*)$', g)
        coeff = 1.0
        body = g
        if m:
            coeff = float(m.group(1))
            body = m.group(2)
        c = parse_group(body)
        for el, ct in c.items():
            total[el] += coeff * ct

    return dict(total)

def element_mass_fractions(formula: str) -> dict:
   
    mix = parse_ti_ca_mass_mix(formula)

    if mix is not None:
        w_ti, w_ca = mix

        frac_tio2  = element_mass_fractions("TiO2")
        frac_caco3 = element_mass_fractions("CaCO3")

        out = collections.Counter()

        for el, frac in frac_tio2.items():
            out[el] += w_ti * frac

        for el, frac in frac_caco3.items():
            out[el] += w_ca * frac

        return dict(out)
    stoich = parse_formula(formula)
    molar_mass = 0.0
    for el, n in stoich.items():
        molar_mass += atomic_mass(el) * n
    el_mf = {}
    for el, n in stoich.items():
        el_mf[el] = atomic_mass(el) * n / molar_mass
    return el_mf

# ============================================================
#  Compound density lookup
# ============================================================

import re

TI_CA_MIX_PATTERN = re.compile(
    r"\s*([0-9.]+)\s*TiO2\s*\+\s*([0-9.]+)\s*CaCO3\s*"
)


def parse_ti_ca_mass_mix(formula: str):
    """Parse a TiO2/CaCO3 mixture specified by mass fractions. Return normalized (w_tio2, w_caco3), or None when the formula does not match."""
    m = TI_CA_MIX_PATTERN.fullmatch(formula)

    if m is None:
        return None

    w_ti = float(m.group(1))
    w_ca = float(m.group(2))

    total = w_ti + w_ca

    if total <= 0.0:
        raise ValueError(
            f"Invalid Mixture TiO2+CaCO3: {formula}"
        )

    
    w_ti /= total
    w_ca /= total

    return w_ti, w_ca

def get_density(formula: str) -> float:
    """Return the compound density in g/cm^3 from the density database, including supported TiO2/CaCO3 mixtures."""

    if formula in density_data:
        return float(density_data[formula]["density_g_cm3"])

    
    mix = parse_ti_ca_mass_mix(formula)

    if mix is not None:
        w_ti, w_ca = mix
    
        rho_ti = float(density_data["TiO2"]["density_g_cm3"])
        rho_ca = float(density_data["CaCO3"]["density_g_cm3"])
    
        
        rho_mix = 1.0 / (
            w_ti / rho_ti
            +
            w_ca / rho_ca
        )
    
        return rho_mix

    raise KeyError(f"Densità non trovata per la formula '{formula}' nel file {DENSITY_FILE}")

# ============================================================
#  Dirichlet compositions per pigment
# ============================================================

def sample_pigment_composition(pigment_components: list, c: float, rng) -> dict:
    """Sample pigment phase mass fractions from a Dirichlet distribution centered on the specified composition ranges. Return a mapping of phase names to normalized mass fractions."""
    
    mu_raw = []
    names  = []
    for comp in pigment_components:
        lo = comp["wt_percent_min"]
        hi = comp["wt_percent_max"]
        mu = (lo + hi) / 2.0
        mu_raw.append(mu)
        names.append(comp["name"])

    mu_raw = np.array(mu_raw, dtype=float)
   
    if np.all(mu_raw == 0):
        w = np.zeros_like(mu_raw)
        w[0] = 1.0
        return {names[i]: w[i] for i in range(len(names))}

    mu_norm = mu_raw / mu_raw.sum()
    alphas = c * mu_norm
    w = rng.dirichlet(alphas)
    return {names[i]: w[i] for i in range(len(names))}

# ============================================================
#  binder-to-pigment ratio (Beta distribution sampling)
# ============================================================

def sample_oil_pigment_ratio(r_min: float, r_max: float, a: float, b: float, rng) -> float:
    """Sample the binder-to-pigment mass ratio within [r_min, r_max] using a linearly rescaled Beta(a, b) distribution."""
    x = rng.beta(a, b)
    r = r_min + (r_max - r_min) * x
    return r

# ============================================================
#  Elemental composition per layer
# ============================================================

OIL_FORMULA = "C57H98O6"  # linseed oil approx.

def compute_elemental_composition_layer(pigment_components, phase_mass_fractions, m_pig, m_oil) -> dict:
    """Compute normalized elemental mass fractions for a layer, including pigment phases and linseed-oil binder."""
    el_mass = collections.Counter()

    
    for comp in pigment_components:
        name = comp["name"]
        formula = comp["formula"]
        w_phase = phase_mass_fractions.get(name, 0.0)  # mass fraction within the pigment
        if w_phase <= 0:
            continue

        try:
            el_frac = element_mass_fractions(formula)
        except Exception as e:
            print(f"[ATTENZIONE] salto composizione elementare per '{name}' ({formula}): {e}")
            continue

        
        m_phase_layer = m_pig * w_phase
        for el, f in el_frac.items():
            el_mass[el] += m_phase_layer * f

    # --- oil contribution (linseed oil ~ C57H98O6) ---
    if m_oil > 0:
        try:
            el_frac_oil = element_mass_fractions(OIL_FORMULA)
            for el, f in el_frac_oil.items():
                el_mass[el] += m_oil * f
        except Exception as e:
            print(f"[ATTENZIONE] salto composizione elementare olio ({OIL_FORMULA}): {e}")

    
    total = sum(el_mass.values())
    if total == 0:
        return {}

    return {el: m / total for el, m in el_mass.items()}


def compute_elemental_composition_pigment_only(pigment_components, phase_mass_fractions) -> dict:
    """Compute normalized elemental mass fractions of the pigment phases, excluding binder. Phase fractions are assumed to sum to one."""
    el_mass = collections.Counter()

    for comp in pigment_components:
        name = comp["name"]
        formula = comp["formula"]
        w_phase = phase_mass_fractions.get(name, 0.0)
        if w_phase <= 0:
            continue

        try:
            el_frac = element_mass_fractions(formula)  # mass fractions within the phase
        except Exception as e:
            print(f"[ATTENZIONE] salto composizione elementare per '{name}' ({formula}): {e}")
            continue

        # the total pigment mass is normalized to one,
        # so the mass of this phase is simply w_phase
        for el, f in el_frac.items():
            el_mass[el] += w_phase * f

    total = sum(el_mass.values())
    if total == 0:
        return {}

    return {el: m / total for el, m in el_mass.items()}

# ============================================================
#  Layer density (pigment and binder)
# ============================================================

def compute_layer_density(pigment_components, phase_mass_fractions, m_pig, m_oil, rho_oil: float) -> float:
    """Compute the density of a pigment-and-binder layer using additive phase volumes and the supplied phase mass fractions and densities."""
    V_tot = 0.0

    # pigment contribution
    for comp in pigment_components:
        name = comp["name"]
        formula = comp["formula"]
        w_phase = phase_mass_fractions.get(name, 0.0)
        if w_phase <= 0:
            continue
        rho_phase = get_density(formula)
        m_phase = m_pig * w_phase
        V_tot += m_phase / rho_phase

    # oil contribution
    V_tot += m_oil / rho_oil

    if V_tot <= 0:
        return np.nan

    rho_layer = 1.0 / V_tot
    return rho_layer

# ============================================================
#  Layer thickness sampling based on pigment fraction
# ============================================================

def sample_thickness(m_pig: float, rng) -> float:
    """Sample effective XRF layer thickness in micrometers from a clipped lognormal distribution. The distribution depends weakly on pigment mass fraction and reflects the assumed mock-up thickness ranges."""

    if m_pig < 0.3:
        # glaze / binder-rich layer
        median_um = 18.0
        sigma = 0.45
        low, high = 2.0, 60.0

    elif m_pig < 0.6:
        # intermediate layer
        median_um = 25.0
        sigma = 0.45
        low, high = 4.0, 90.0

    else:
        # pigment-rich layer / impasto, with a moderate thickness cap
        median_um = 32.0
        sigma = 0.50
        low, high = 6.0, 110.0

    t = rng.lognormal(mean=np.log(median_um), sigma=sigma)
    return float(np.clip(t, low, high))

# ============================================================
#  Ground layer selection
# ============================================================
"""
SUPPORT_OPTIONS = [
    ("Lead white",     "2PbCO3·Pb(OH)2", 0.2),
    ("Titanium white", "TiO2",           0.2),
    ("Calcite",        "CaCO3",          0.2),
    ("Gypsum",         "CaSO4·2H2O",     0.2),
    ("Ti+Ca acrylic support", "0.7TiO2+0.3CaCO3", 0.2)
    
]

def sample_support(rng):
   
    names = [x[0] for x in SUPPORT_OPTIONS]
    formulas = [x[1] for x in SUPPORT_OPTIONS]
    probs = [x[2] for x in SUPPORT_OPTIONS]
    idx = rng.choice(len(names), p=probs)
    return names[idx], formulas[idx]
"""

SUPPORT_OPTIONS = [
    ("Lead white",     "2PbCO3·Pb(OH)2", 0.1),
    ("Titanium white", "TiO2",           0.5),
    ("Calcite",        "CaCO3",          0.2),
    ("Gypsum",         "CaSO4·2H2O",     0.2),
    #("Ti+Ca acrylic support", None,      0.4)
]


def sample_ti_ca_support(
    rng,
    ti_center=0.58,
    ti_sigma=0.07,
    ti_min=0.45,
    ti_max=0.70,
):
    """Sample the inorganic composition of a TiO2/CaCO3 ground layer. The TiO2 fraction is drawn from a clipped normal distribution; binder dilution is handled separately."""

    c_ti = float(
        np.clip(
            rng.normal(
                loc=ti_center,
                scale=ti_sigma,
            ),
            ti_min,
            ti_max,
        )
    )

    c_ca = float(
        1.0
        - c_ti
    )

    formula = (
        f"{c_ti:.3f}TiO2+"
        f"{c_ca:.3f}CaCO3"
    )

    name = (
        "Ti+Ca acrylic support "
        f"({c_ti:.2f} TiO2, {c_ca:.2f} CaCO3)"
    )

    return name, formula


def sample_support(rng):
    names = [x[0] for x in SUPPORT_OPTIONS]
    formulas = [x[1] for x in SUPPORT_OPTIONS]
    probs = [x[2] for x in SUPPORT_OPTIONS]

    idx = rng.choice(len(names), p=probs)

    name = names[idx]
    formula = formulas[idx]

    if name == "Ti+Ca acrylic support":
        return sample_ti_ca_support(rng)

    return name, formula
    
# ============================================================
#  Simulate one paint layer
# ============================================================

def simulate_paint_layer(layer_index: int, rng, pigment_name=None):
    """Generate one paint layer: select a pigment, sample its phases and binder fraction, and compute density, thickness, and elemental composition."""
    SUPPORT_ONLY = {"Lead White", "Titanium White", "Calcite", "Gypsum","San Giovanni white"}

    # pigments allowed in paint layers
    allowed_pigments = [p for p in pigment_names if p not in SUPPORT_ONLY]

    if pigment_name is None:
        # randomly select an allowed pigment
        pigment_name = rng.choice(allowed_pigments)
    else:
        # replace any ground-only pigment with an allowed paint pigment
        if pigment_name in SUPPORT_ONLY:
            # optional diagnostic logging
            # print(f"[simulate_paint_layer] '{pigment_name}' is ground-only; selecting another paint pigment.")
            pigment_name = rng.choice(allowed_pigments)


    pigment = pigment_data[pigment_name]
    components = pigment["components"]

    # pigment phase composition
    phase_mass_fractions = sample_pigment_composition(components, C_DIRICHLET, rng)

    # binder-to-pigment mass ratio and corresponding mass fractions
    r = sample_oil_pigment_ratio(R_MIN, R_MAX, BETA_A, BETA_B, rng)
    m_pig = 1.0 / (1.0 + r)
    m_oil = r   / (1.0 + r)

    # layer density
    rho_layer = compute_layer_density(components, phase_mass_fractions, m_pig, m_oil, RHO_OIL)

    # layer thickness
    thickness = sample_thickness(m_pig, rng)

    # elemental composition
    elem_comp = compute_elemental_composition_layer(components, phase_mass_fractions, m_pig, m_oil)

    return {
        "layer_index": layer_index,
        "type": "paint",
        "pigment_name": pigment_name,
        "phase_mass_fractions": phase_mass_fractions,
        "m_pig": m_pig,
        "m_oil": m_oil,
        "oil_to_pig_ratio": r,
        "rho_layer_g_cm3": rho_layer,
        "thickness_um": thickness,
        "elemental_composition": elem_comp
    }

# ============================================================
#  Simulate a complete paint stratigraphy (paint layers and ground)
# ============================================================

def simulate_painting(painting_id: int, rng):
    """Generate a complete paint stratigraphy with one or two paint layers plus a ground layer. Paint-layer count follows a capped Poisson distribution."""
    # Poisson-distributed paint layer count, capped below
    N_layers = max(1, rng.poisson(LAMBDA_LAYERS))
    N_layers = min(N_layers, 2)

    layers = []

    # shuffled list of available pigments
    available_pigments = pigment_names.copy()
    rng.shuffle(available_pigments)

    for i in range(N_layers):
        if not available_pigments:
            break  # stop if no unused pigments remain

        this_pigment = available_pigments.pop()  # take a pigment and remove it from the pool

        layer_info = simulate_paint_layer(
            layer_index=i+1,
            rng=rng,
            pigment_name=this_pigment
        )
        layers.append(layer_info)

    # --- ground layer parameters ---
    support_name, support_formula = sample_support(rng)

    m_pig_sup = 0.32
    m_oil_sup = 0.68
    r_support = 0.68/0.32

    phase_mass_fractions_sup = {support_name: 1.0}

    rho_support_layer = compute_layer_density(
        [{"name": support_name, "formula": support_formula}],
        phase_mass_fractions_sup,
        m_pig_sup,
        m_oil_sup,
        RHO_OIL
    )

    elem_comp_support = compute_elemental_composition_layer(
        [{"name": support_name, "formula": support_formula}],
        phase_mass_fractions_sup,
        m_pig=m_pig_sup,
        m_oil=m_oil_sup
    )

    support_layer = {
        "layer_index": len(layers) + 1,   # actual number of paint layers plus one
        "type": "support",
        "pigment_name": support_name,
        "phase_mass_fractions": phase_mass_fractions_sup,
        "m_pig": m_pig_sup,
        "m_oil": m_oil_sup,
        "oil_to_pig_ratio": r_support,
        "rho_layer_g_cm3": rho_support_layer,
        "thickness_um": float(np.clip(
            rng.lognormal(mean=np.log(32.0), sigma=0.60),
            5.0,
            130.0
        )),
        "elemental_composition": elem_comp_support
    }

    layers.append(support_layer)

    return layers



# In[8]:


def build_spectrum_input_from_painting(painting,
                                       include_support=True,
                                       skip_empty_layers=True):
    """Convert a simulated stratigraphy into XRF model inputs: elements, fluorescence series, layer densities, thicknesses in cm, and elemental mass fractions."""

    # special fluorescence-series assignments
    special_series = {
        "Pb": ["L", "M"],
        "Hg": ["L", "M"],
        "Cd": ["L", "K"],
        # add other elements here if needed: "Au": ["L", "M"], ...
    }

    el_layers = []
    series_layers = []
    rho = []
    t_layers = []
    W_ml = []

    for layer in painting:
        if layer["type"] == "support" and not include_support:
            continue

        comp = layer.get("elemental_composition", {})
        if (not comp) and skip_empty_layers:
            continue

        rho_layer = layer.get("rho_layer_g_cm3", np.nan)
        thickness_um = layer.get("thickness_um", np.nan)
        if np.isnan(rho_layer) or np.isnan(thickness_um):
            # skip invalid layers
            if skip_empty_layers:
                continue

        # density and thickness
        rho.append(rho_layer)
        t_layers.append(thickness_um * 1e-4)  # µm -> cm

        this_el = []
        this_series = []
        this_W = []

        # comp: {element: mass_fraction}
        for el, w in comp.items():
            # select fluorescence series for this element
            series_for_el = special_series.get(el, ["K"])
            for s in series_for_el:
                this_el.append(el)
                this_series.append(s)
                this_W.append(w)   # repeat the same mass fraction for each series

        el_layers.append(this_el)
        series_layers.append(this_series)
        W_ml.append(this_W)

    t_layers = np.array(t_layers, dtype=float)

    return el_layers, series_layers, rho, t_layers, W_ml


# In[1]:


import os
import json
import numpy as np
import matplotlib.pyplot as plt
import xraydb

from functools import lru_cache
from scipy.special import exp1, wofz

# =========================
# Atomic-number cutoff
# =========================
Z_MIN_INTENSITY = xraydb.atomic_number("Al")   # = 13

# =========================
# Cached wrappers
# =========================

@lru_cache(maxsize=None)
def atomic_number_cached(element: str) -> int:
    return xraydb.atomic_number(element)

@lru_cache(maxsize=None)
def mu_elam_cached(element: str, E_eV: float, kind: str = 'tot') -> float:
    # mu_elam must be defined or imported
    return mu_elam(element, E_eV, kind=kind)

@lru_cache(maxsize=None)
def xray_edge_cached(element: str, edge: str):
    return xraydb.xray_edge(element, edge)

@lru_cache(maxsize=None)
def fluor_yield_cached(element: str, series_subshell: str, line: str, E_eV: float):
    # fluor_yield must be defined or imported
    return fluor_yield(element, series_subshell, line, E_eV)

@lru_cache(maxsize=None)
def ck_prob_cached(Z: int, src: str, dest: str) -> float:
    # ck_probability must be defined or imported
    return ck_probability(Z, src, dest)


# =========================
# 1. chi, D, L0
# =========================

def chi_(mu, mu_i, phi):
    """Compute the combined incident/emission attenuation term chi."""
    return mu/np.sin(phi[0]) + mu_i/np.sin(phi[1])


def D(x):
    """Evaluate the de Boer auxiliary function D(x) = exp(x) * E1(x), including numerical edge cases."""
    x = float(x)
    if x == 0 or abs(x) > 709:
        return 0.0
    if x > 0.0:
        return float(np.exp(x) * exp1(x))
    else:
        val = np.exp(x) * exp1(complex(x, 0.0))
        return float(val.real)


def L0(mu1, mu2, muj, t):
    """Evaluate the L0 auxiliary term used in intra-layer secondary fluorescence."""
    pre = (mu1 + mu2) / (1.0 - np.exp(-(mu1 + mu2)*t))

    par = (
        -D(muj*t)             / (mu1*mu2)
        + D((muj - mu2)*t)    / (mu2*(mu1 + mu2))
        + D((mu1 + muj)*t)    / (mu1*(mu1 + mu2))
    )

    add = (
        1.0/(mu1*(mu1 + mu2)) * np.log(abs(1.0 + mu1/muj))
        + np.exp(-(mu1 + mu2)*t)/(mu2*(mu1 + mu2))
          * np.log(abs(1.0 - mu2/muj))
    )

    return pre * (np.exp(-(mu1 + muj)*t)*par + add)


# =========================
# 2. set_line
# =========================

def set_line(series):
    """Map a fluorescence series to its modeled emission lines and base subshell."""
    if series == 'K':
        lines = ['Ka', 'Kb']
        series_subshell = 'K'
    elif 'L' in series:
        lines = ['La', 'Lb', 'Lg','Lb2']
        series_subshell = 'L'
    elif 'M5' in series:
        lines = ['Ma']
        series_subshell = 'M5'
    else:
        lines = []
        series_subshell = series
    return lines, series_subshell


# =========================
# 3. Optimized subshell excitation probability
# =========================

def prob_line(element, series_subshell):
    """Calculate subshell excitation probability, including Coster–Kronig contributions where applicable, using cached atomic data."""
    Z = atomic_number_cached(element)

    edge = xray_edge_cached(element, series_subshell)
    jump = edge[2]

    if series_subshell in ("K", "L1"):
        prob = (1.0 - 1.0/jump)

    elif series_subshell == "L2":
        jump_1 = xray_edge_cached(element, "L1")[2]
        prob = ((1.0 - 1.0/jump)
                + (1.0 - 1.0/jump_1)*ck_prob_cached(Z, "L1", "L2"))

    elif series_subshell == "L3":
        jump_1 = xray_edge_cached(element, "L1")[2]
        jump_2 = xray_edge_cached(element, "L2")[2]
        prob = ((1.0 - 1.0/jump)
                + (1.0 - 1.0/jump_2)*ck_prob_cached(Z, "L2", "L3")
                + (1.0 - 1.0/jump_1)*ck_prob_cached(Z, "L1", "L3"))

    elif series_subshell == "M5":
        jump_1 = xray_edge_cached(element, "M1")[2]
        jump_2 = xray_edge_cached(element, "M2")[2]
        jump_3 = xray_edge_cached(element, "M3")[2]
        jump_4 = xray_edge_cached(element, "M4")[2]
        prob = (
            (1.0 - 1.0/jump)
            + (1.0 - 1.0/jump_2)*ck_prob_cached(Z, "M2", "M5")
            + (1.0 - 1.0/jump_1)*ck_prob_cached(Z, "M1", "M5")
            + (1.0 - 1.0/jump_3)*ck_prob_cached(Z, "M3", "M5")
            + (1.0 - 1.0/jump_4)*ck_prob_cached(Z, "M4", "M5")
        )
    else:
        prob = 0.0

    return prob


# =========================
# 4. S_intra (intra-layer)
# =========================

def S_intra(el_i, el_j, phi1, phi2, seriesi, linei, seriesj,
            W, elements, E0, t):
    """Compute intra-layer secondary-fluorescence enhancement from a source element j to a target line of element i."""
    corr = []

    # absorption edge of target element i
    edge_i_E = xray_edge_cached(el_i, seriesi)[0]

    # emission energy of element i (used for mu_li)
    fy_i = fluor_yield_cached(el_i, seriesi, linei, E0)
    E_line_i = fy_i[1]

    lines, base_series_j = set_line(seriesj)

    for linej in lines:
        # map line labels to the actual subshell of j
        ss_j = base_series_j
        if linej in ['Lb', 'Lg']:
            ss_j = 'L2'
        elif linej in ['La','Lb2']:
            ss_j = 'L3'
        elif linej == 'Ma':
            ss_j = 'M5'

        fy_j = fluor_yield_cached(el_j, ss_j, linej, E0)
        E_line_j = fy_j[1]

        # threshold condition: emission from j must excite i
        if E_line_j < edge_i_E:
            corr.append(0.0)
            continue

        # calculate attenuation at incident and emitted energies
        mu   = 0.0
        mulj = 0.0
        muli = 0.0
        seen = set()

        for elem, w in zip(elements, W):
            if elem in seen:
                continue
            seen.add(elem)
            mu   += mu_elam_cached(elem, E0,         kind='tot') * w
            mulj += mu_elam_cached(elem, E_line_j,   kind='tot') * w
            muli += mu_elam_cached(elem, E_line_i,   kind='tot') * w

        # photoelectric cross-section factors
        tauj   = mu_elam_cached(el_j, E0,       kind='photo')
        tauilj = mu_elam_cached(el_i, E_line_j, kind='photo')
        taui   = mu_elam_cached(el_i, E0,       kind='photo')

        # q_j factor
        yield_j  = fy_j[0]
        E_line_j = fy_j[1]
        
        # the primary beam must ionize the source subshell of j
        if yield_j <= 0.0:
            corr.append(0.0)
            continue
        
        # the emitted line from j must ionize target i
        if E_line_j < edge_i_E:
            corr.append(0.0)
            continue
        
        qj = (
            yield_j
            * fy_j[2]
            * prob_line(
                el_j,
                ss_j,
            )
        )

        chi1 = mu  / np.sin(phi1)
        chi2 = muli/ np.sin(phi2)

        Sij2 = 0.5 * qj * tauj * tauilj / taui * (
          L0(chi1, chi2, mulj, t) + L0(chi2, chi1, mulj, t)
        )

        # numerical clamp: negative enhancement is nonphysical
        if Sij2 < 0.0:
            if Sij2 < -1e-3:
                print("ATTENZIONE Sij2 molto negativo:", Sij2, "per", el_i, el_j)
            Sij2 = 0.0

        corr.append(Sij2)

    return corr


# =========================
# 5. I_line_t with atomic-number cutoff Z >= Al
# =========================

def I_line_t(elements, series, E0, phi, W, A, d, t, enance):
    """Calculate fluorescence line intensities for a single layer, optionally including intra-layer enhancement. Only elements with Z >= Al emit modeled lines; all elements contribute to attenuation."""
    I_i = []
    l2  = []

    domega = A / d**2

    # total layer attenuation at E0, including light elements
    mu = 0.0
    seen_mu = set()
    for el, w in zip(elements, W):
        if el in seen_mu:
            continue
        seen_mu.add(el)
        mu += mu_elam_cached(el, E0, kind='tot') * w

    # iterate over elements
    for idx_el, el in enumerate(elements):
        Z_el = atomic_number_cached(el)
        tau = mu_elam_cached(el, E0, kind='photo')

        lines, base_series = set_line(series[idx_el])

        for line in lines:
            # map line labels to the actual subshell
            series_subshell = base_series
            if line in ['Lb', 'Lg']:
                series_subshell = 'L2'
            elif line in ['La','Lb2']:
                series_subshell = 'L3'
            elif line == 'Ma':
                series_subshell = 'M5'

            # emission-line energy
            fy_i = fluor_yield_cached(el, series_subshell, line, E0)
            E_line = fy_i[1]

            # subshell absorption edge
            edge_data = xray_edge_cached(el, series_subshell)
            edge_E = edge_data[0]

            # initialize line intensity to zero
            I_line = 0.0

            # conditions for computing emission intensity:
            #   - E0 > edge
            #   - Z >= Z_MIN_INTENSITY (e.g., Al)
            if (E0 > edge_E) and (Z_el >= Z_MIN_INTENSITY):

                # total attenuation at the emitted line energy
                mu_i = 0.0
                seen_mu_i = set()
                for el2, w2 in zip(elements, W):
                    if el2 in seen_mu_i:
                        continue
                    seen_mu_i.add(el2)
                    mu_i += mu_elam_cached(el2, E_line, kind='tot') * w2

                chi_val = chi_(mu, mu_i, phi)

                # q factor
                q = edge_data[1] * fy_i[2] * prob_line(el, series_subshell)

                I_base = (domega/(4*np.pi*np.sin(phi[0]))) * q * tau/chi_val * W[idx_el]
                I_base *= (1.0 - np.exp(-chi_val * t))

                if enance:
                    # enhancement intra-layer
                    S2 = []
                    for el_j, series_j in zip(elements, series):
                        S_ij_list = S_intra(
                            el, el_j, phi[0], phi[1],
                            series_subshell, line, series_j,
                            W, elements, E0, t
                        )
                        S_ij = np.sum(S_ij_list)
                        S2.append(S_ij)

                    Stot = 0.0
                    for w_j, S_ij in zip(W, S2):
                        Stot += S_ij * w_j

                    I_line = I_base * (1.0 + Stot)
                else:
                    I_line = I_base

            # append zero intensities too, preserving line ordering
            I_i.append(I_line)
            l2.append(E_line)

    return I_i, l2


# ==================
# Line profiles
# ==================
LN2 = np.log(2.0)

def gaussian_fwhm_eV_from_keV(E_keV: float) -> float:
    """Return the assumed instrumental Gaussian FWHM (eV) as a function of energy (keV)."""
    return float(np.sqrt(31.3**2 + 2576.84 * E_keV))

def fwhm_to_sigma(fwhm):
    return fwhm / (2.0 * np.sqrt(2.0 * LN2))

def gaussian_profile(x, mu, fwhm_G):
    sigma = fwhm_to_sigma(fwhm_G)
    return np.exp(-0.5 * ((x - mu)/sigma)**2) / (sigma * np.sqrt(2*np.pi))

def lorentz_profile(x, mu, fwhm_L):
    gamma = 0.5 * fwhm_L
    return (gamma/np.pi) / ((x - mu)**2 + gamma**2)

def voigt_profile(x, mu, fwhm_G, fwhm_L):
    sigma = fwhm_to_sigma(fwhm_G)
    gamma = 0.5 * fwhm_L
    if sigma <= 0:
        return lorentz_profile(x, mu, fwhm_L)
    z = (x - mu + 1j*gamma) / (sigma * np.sqrt(2))
    return np.real(wofz(z)) / (sigma * np.sqrt(2*np.pi))


# ==================
# Detection efficiency and line widths
# ==================

@lru_cache(maxsize=None)
def load_common_dict_raw(json_path: str) -> dict:
    """Load the line-efficiency and linewidth JSON database, caching the result by file path."""
    with open(json_path, "r") as f:
        return json.load(f)


def _eff_and_width_from_common(common: dict, element: str, Epk_eV: float,
                               default_eff: float = 1.0,
                               default_width: float = 1.5):
    """Look up detector efficiency and natural Lorentzian FWHM for the nearest emission line in the common XRF database."""

    entry = common.get(element, None)
    if entry is None:
        return default_eff, default_width

    eff_list = entry.get("efficiency", [])
    widths_dict = entry.get("line_widths", {})

    eff_val = default_eff
    line_name = None

    # --- efficiency and line label ---
    if isinstance(eff_list, list) and len(eff_list) > 0:
        E_keV = Epk_eV / 1000.0
        energies_keV = np.array([float(e["Energy_keV"]) for e in eff_list], dtype=float)
        idx = int(np.argmin(np.abs(energies_keV - E_keV)))
        eff_entry = eff_list[idx]
        eff_val = float(eff_entry.get("Efficiency", default_eff))
        line_name = eff_entry.get("Line", None)

    # --- natural linewidth ---
    fwhm_L = None
    if isinstance(widths_dict, dict) and len(widths_dict) > 0:
        # 1) first try an exact line-name match
        if line_name is not None and line_name in widths_dict:
            fwhm_L = float(widths_dict[line_name])
        # 2) otherwise try the line family (Ka, Kb, La, Lb, Ma, ...)
        elif line_name is not None and len(line_name) >= 2:
            fam = line_name[:2]  # es. "Ka3" -> "Ka"
            candidate_keys = [k for k in widths_dict.keys() if k.startswith(fam)]
            if candidate_keys:
                # use the first matching entry (typically 'Ka1', 'Kb1', ...)
                fwhm_L = float(widths_dict[candidate_keys[0]])
        # 3) fallback: first dictionary entry
        if fwhm_L is None:
            first_key = next(iter(widths_dict.keys()))
            fwhm_L = float(widths_dict[first_key])

    if fwhm_L is None:
        fwhm_L = default_width

    return eff_val, fwhm_L


# ==================================
# Inter-layer upward and downward enhancement terms
# ==================================

def M_up(mu1l, mu2l, mu1i, mu1j, mubj, mubl, mu2j, t1, t2, tb, phi):

    def X(p, q, d1, d2):

        def V(d1_local, d2_local):
            if d1_local == 0 and d2_local == 0:
                V_val = -1/(p*mu1j*np.sin(phi[1]) + q*mu2j*np.sin(phi[1])) * (
                    mu2j*np.sin(phi[1])/p*np.log(abs(1 + p/(mu2j*np.sin(phi[1])))) +
                    mu1j*np.sin(phi[1])/q*np.log(abs(1 - q/(mu1j*np.sin(phi[1]))))
                )
            elif d1_local == np.inf or d2_local == np.inf:
                V_val = 0
            else:

                pre  = np.exp((q - mu1j*np.sin(phi[1]))*d1_local
                              - (p + mu2j*np.sin(phi[1]))*d2_local
                              - mubj*np.sin(phi[1])*tb)
                arg  = (mu1j*np.sin(phi[1])*d1_local
                        + mubj*np.sin(phi[1])*tb
                        + mu2j*np.sin(phi[1])*d2_local)

                fac1 = D((1 + p/(mu2j*np.sin(phi[1])))*arg) * mu2j*np.sin(phi[1]) / (
                    p*(p*mu1j*np.sin(phi[1]) + q*mu2j*np.sin(phi[1]))
                )
                fac2 = D((1 - q/(mu1j*np.sin(phi[1])))*arg) * mu1j*np.sin(phi[1]) / (
                    q*(p*mu1j*np.sin(phi[1]) + q*mu2j*np.sin(phi[1]))
                )
                fac3 = D(arg) / (p*q)
                V_val = pre*(fac1 + fac2 - fac3)
            return V_val

        return V(d1, d2) - V(d1, 0.0) - V(0.0, d2) + V(0.0, 0.0)

    M_val = (mu1l + mu1i)/(np.exp((mu1l + mu1i)*t1) - 1.0) * np.exp(-mubl*tb) * X(mu2l, mu1i, t1, t2)
    return M_val


def M_down(mu1l, mu2l, mu1i, mu1j, mubj, mubl, mu2j, t1, t2, tb, phi):

    def X(p, q, d1, d2):

        def V(d1_local, d2_local):
            if d1_local == 0 and d2_local == 0:
                V_val = -1/(p*mu1j*np.sin(phi[1]) + q*mu2j*np.sin(phi[1])) * (
                    mu2j*np.sin(phi[1])/p*np.log(abs(1 + p/(mu2j*np.sin(phi[1])))) +
                    mu1j*np.sin(phi[1])/q*np.log(abs(1 - q/(mu1j*np.sin(phi[1]))))
                )
            elif d1_local == np.inf or d2_local == np.inf:
                V_val = 0
            else:

                pre  = np.exp((q - mu1j*np.sin(phi[1]))*d1_local
                              - (p + mu2j*np.sin(phi[1]))*d2_local
                              - mubj*np.sin(phi[1])*tb)
                arg  = (mu1j*np.sin(phi[1])*d1_local
                        + mubj*np.sin(phi[1])*tb
                        + mu2j*np.sin(phi[1])*d2_local)

                fac1 = D((1 + p/(mu2j*np.sin(phi[1])))*arg) * mu2j*np.sin(phi[1]) / (
                    p*(p*mu1j*np.sin(phi[1]) + q*mu2j*np.sin(phi[1]))
                )
                fac2 = D((1 - q/(mu1j*np.sin(phi[1])))*arg) * mu1j*np.sin(phi[1]) / (
                    q*(p*mu1j*np.sin(phi[1]) + q*mu2j*np.sin(phi[1]))
                )
                fac3 = D(arg) / (p*q)
                V_val = pre*(fac1 + fac2 - fac3)
            return V_val

        return V(d1, d2) - V(d1, 0.0) - V(0.0, d2) + V(0.0, 0.0)

    M_val = (mu1l + mu1i)/(1.0 - np.exp(-(mu1l + mu1i)*t1)) * np.exp(-mubl*tb) * X(mu1i, mu2l, t2, t1)
    return M_val


# =========================
# μ_eff e _mu_layer_sum
# =========================

def _mu_layer_sum(el_layer, W_layer, E_eV, kind='tot'):
    """Return the mass-fraction-weighted attenuation coefficient of a layer, without geometric angle factors."""
    mu_val = 0.0
    seen = set()
    for el, w in zip(el_layer, W_layer):
        if el in seen:
            continue
        seen.add(el)
        mu_val += mu_elam_cached(el, E_eV, kind=kind) * w
    return mu_val


def mu_eff(el_layers, W_layers, layer_index, E_eV, is_primary, phi):
    """Return the effective attenuation coefficient corrected for the incident or emission angle."""
    mu = _mu_layer_sum(el_layers[layer_index], W_layers[layer_index], E_eV, kind='tot')
    sin_in, sin_out = np.sin(phi[0]), np.sin(phi[1])
    return mu / (sin_in if is_primary else sin_out)


# =========================
# list_lines_for_layer
# =========================

def list_lines_for_layer(elements, series_list, E0):
    """Enumerate emission-line energies and metadata in the same order as I_line_t."""
    energies = []
    meta = []
    for el, ser in zip(elements, series_list):
        lines, series_subshell = set_line(ser)
        for line in lines:
            ss = series_subshell
            if line in ['Lb', 'Lg']:
                ss = 'L2'
            elif line in ['La','Lb2']:
                ss = 'L3'
            elif line == 'Ma':
                ss = 'M5'
            fy = fluor_yield_cached(el, ss, line, E0)
            E_line = fy[1]
            energies.append(E_line)
            meta.append({'element': el, 'series_subshell': ss, 'line': line})
    return np.array(energies, dtype=float), meta


# =========================
# S_inter_for_line (inter-layer)
# =========================


def get_element_weight_in_layer(el_layer, W_layer, target_el):
    """Retrieve the mass fraction of an element in a layer, checking consistency when it appears in multiple fluorescence series."""
    values = [
        float(w)
        for el, w in zip(el_layer, W_layer)
        if el == target_el
    ]

    if not values:
        return 0.0

    if not np.allclose(values, values[0], rtol=0.0, atol=1e-12):
        raise ValueError(
            f"Pesi inconsistenti per {target_el}: {values}"
        )

    return values[0]

def S_inter_for_line(
        k, idx_i,
        I_i_intra, E_i,
        meta_i,
        el_layers, series_layers, W_layers, t_layers,
        phi, E0,
        layer_lines_cache=None,
        mu_lambda_layers=None
    ):
    """Calculate secondary-fluorescence enhancement of one target line due to other layers, including intervening-layer attenuation."""
    n_layers = len(el_layers)
    el_i = meta_i['element']
    seriesi = meta_i['series_subshell']

    contributions = []
    S_tot = 0.0

    EPS_WARN = 1e-3

    edge_i_E = xray_edge_cached(el_i, seriesi)[0]


    for L_j in range(n_layers):
        if L_j == k:
            continue

        # identify the layers and intervening thicknesses
        if L_j > k:
            L1 = k
            L2 = L_j
            d1 = t_layers[L1]
            d2 = t_layers[L2]
            Lb_list = list(range(L1+1, L2))
        else:
            L1 = L_j
            L2 = k
            d1 = t_layers[L2]
            d2 = t_layers[L1]
            Lb_list = list(range(L1+1, L2))

        db = sum(t_layers[m] for m in Lb_list) if Lb_list else 0.0

        # use cached emission lines for layer j when available
        if layer_lines_cache is not None:
            E_lines_j, meta_j_list = layer_lines_cache[L_j]
        else:
            elements_j = el_layers[L_j]
            series_j   = series_layers[L_j]
            E_lines_j, meta_j_list = list_lines_for_layer(elements_j, series_j, E0)

        for meta_j, E_j in zip(meta_j_list, E_lines_j):
            el_j   = meta_j['element']
            linej  = meta_j['line']
            seriesj = meta_j['series_subshell']

            w_j = get_element_weight_in_layer(
                el_layers[L_j],
                W_layers[L_j],
                el_j,
            )
            
            if w_j <= 0.0:
                continue

            # threshold: emission from j must excite i
            if E_j < edge_i_E:
                continue

            # tau
            tau_j_lambda = mu_elam_cached(el_j, E0, kind='photo')
            tau_i_lambda = mu_elam_cached(el_i, E0, kind='photo')
            tau_i_j      = mu_elam_cached(el_i, E_j, kind='photo')

            fy_j = fluor_yield_cached(
                el_j,
                seriesj,
                linej,
                E0,
            )
            
            yield_j = fy_j[0]
            
            # the primary beam must ionize the source subshell
            if yield_j <= 0.0:
                continue
            
            qj = (
                yield_j
                * fy_j[2]
                * prob_line(
                    el_j,
                    seriesj,
                )
            )
            # μ_eff
            if mu_lambda_layers is not None:
                mu1l = mu_lambda_layers[k]
                mu2l = mu_lambda_layers[L_j]
                if Lb_list:
                    mub_lambda = sum(mu_lambda_layers[m] for m in Lb_list)
                else:
                    mub_lambda = 0.0
            else:
                mu1l = mu_eff(el_layers, W_layers, k,    E0,  True,  phi)
                mu2l = mu_eff(el_layers, W_layers, L_j,  E0,  True,  phi)
                if Lb_list:
                    mub_lambda = sum(mu_eff(el_layers, W_layers, m, E0, True,  phi) for m in Lb_list)
                else:
                    mub_lambda = 0.0

            mu1i = mu_eff(el_layers, W_layers, k,   E_i, False, phi)
            mu1j = mu_eff(el_layers, W_layers, k,   E_j, False, phi)
            mu2j = mu_eff(el_layers, W_layers, L_j, E_j, False, phi)

            if Lb_list:
                mub_j = sum(mu_eff(el_layers, W_layers, m, E_j, False, phi) for m in Lb_list)
            else:
                mub_j = 0.0

            if L_j > k:
                M_val = M_up(mu1l, mu2l, mu1i, mu1j, mub_j, mub_lambda, mu2j, d1, d2, db, phi)
            else:
                M_val = M_down(mu1l, mu2l, mu1i, mu1j, mub_j, mub_lambda, mu2j, d1, d2, db, phi)

            S_ij_raw = 0.5 * qj * tau_j_lambda * tau_i_j / tau_i_lambda * M_val*w_j

            # numerical clamp: negative enhancement is nonphysical
            S_ij = S_ij_raw
            if S_ij < 0.0:
                if S_ij < -EPS_WARN:
                    pass
                S_ij = 0.0

            if S_ij != 0.0:
                contributions.append((L_j, el_j, S_ij))
                S_tot += S_ij

    # final nonnegative clamp on total enhancement
    if S_tot < 0.0:
        if S_tot < -EPS_WARN:
            pass
        S_tot = 0.0

    return S_tot, contributions


# =========================
# Monochromatic and polychromatic multilayer calculations
# =========================

def I_multil_general_mono(el_layers, series_layers, E0, phi, W_layers, A, d, t_layers,
                          enance=True, verbose=True):
    """Compute the multilayer fluorescence spectrum for one incident photon energy, including attenuation and inter-layer enhancement."""
    n_layer = len(el_layers)
    results_per_layer = []
    I_all, E_all = [], []

    # cache emission lines for inter-layer calculations
    layer_lines_cache = []
    for j in range(n_layer):
        elements_j = el_layers[j]
        series_j   = series_layers[j]
        E_lines_j, meta_j_list = list_lines_for_layer(elements_j, series_j, E0)
        layer_lines_cache.append((E_lines_j, meta_j_list))

    # cache effective incident attenuation per layer
    mu_lambda_layers = [
        mu_eff(el_layers, W_layers, k, E0, True, phi) for k in range(n_layer)
    ]

    for k in range(n_layer):
        elements_k = el_layers[k]
        series_k   = series_layers[k]
        W_k        = W_layers[k]
        t_k        = t_layers[k]

        I_k_intra, E_k_lines = I_line_t(elements_k, series_k, E0, phi, W_k, A, d, t_k, enance)
        I_k_intra = np.asarray(I_k_intra, dtype=float)
        E_k_lines = np.asarray(E_k_lines, dtype=float)

        sin_in, sin_out = np.sin(phi[0]), np.sin(phi[1])

        # incident attenuation is independent of emission-line energy
        mu_in_sum = 0.0
        for m in range(0, k):
            mu_in_sum += _mu_layer_sum(el_layers[m], W_layers[m], E0, kind='tot') * t_layers[m] / sin_in
        A_in = np.exp(-mu_in_sum)
        A_in_arr = np.full_like(E_k_lines, A_in, dtype=float)

        # outgoing attenuation depends on emission-line energy
        A_out_list = []
        for E_line in E_k_lines:
            mu_out_sum = 0.0
            for m in range(0, k):
                mu_out_sum += _mu_layer_sum(el_layers[m], W_layers[m], E_line, kind='tot') * t_layers[m] / sin_out
            A_out_list.append(np.exp(-mu_out_sum))
        A_out_arr = np.asarray(A_out_list)

        I_k_corr_intra = I_k_intra * A_in_arr * A_out_arr

        _, meta_k_list = layer_lines_cache[k]  # same ordering as I_k_intra/E_k_lines
        assert len(meta_k_list) == len(E_k_lines)

        I_k_corr_total = np.copy(I_k_corr_intra)

        if verbose:
            print(f"\n=== Layer {k} ===")

        for idx, (I_i0, E_i, meta_i) in enumerate(zip(I_k_corr_intra, E_k_lines, meta_k_list)):
            S_tot, contribs = S_inter_for_line(
                k, idx, I_i0, E_i, meta_i,
                el_layers, series_layers, W_layers, t_layers,
                phi, E0,
                layer_lines_cache=layer_lines_cache,
                mu_lambda_layers=mu_lambda_layers
            )
            I_new = I_i0 * (1.0 + S_tot)
            I_k_corr_total[idx] = I_new

            if verbose:
                el_i = meta_i['element']
                line_i = meta_i['line']
                print(f"  Riga {idx}: {el_i}({line_i}), E={E_i:.1f} eV")
                print(f"    I_intra_corr = {I_i0:.3e}")
                print(f"    S_inter_tot  = {S_tot:.3e}  -> I_tot = {I_new:.3e}")
                if contribs:
                    for (L_j, el_j, S_ij) in contribs:
                        print(f"      da layer {L_j}, elemento {el_j}: S_ij = {S_ij:.3e}")
                else:
                    print("      nessun contributo inter-layer rilevante")

        results_per_layer.append((I_k_corr_total, E_k_lines))
        I_all.append(I_k_corr_total)
        E_all.append(E_k_lines)

    I_all = np.concatenate(I_all) if I_all else np.array([])
    E_all = np.concatenate(E_all) if E_all else np.array([])

    return results_per_layer, (I_all, E_all)


def I_multil_general(el_layers, series_layers, E0, phi, W_layers, A, d, t_layers,
                     enance=True, spectrum_weights=None, verbose=True, atol_E=1e-3):
    """Compute multilayer fluorescence for a monochromatic source or a weighted discrete incident-energy spectrum."""
    E0_arr = np.atleast_1d(E0).astype(float)

    if E0_arr.size == 1:
        return I_multil_general_mono(
            el_layers, series_layers,
            E0_arr[0], phi, W_layers, A, d, t_layers,
            enance=enance, verbose=verbose
        )

    n_E = E0_arr.size
    if spectrum_weights is None:
        w_arr = np.ones_like(E0_arr, dtype=float)
    else:
        w_arr = np.asarray(spectrum_weights, dtype=float)
        if w_arr.size != n_E:
            raise ValueError("spectrum_weights deve avere la stessa lunghezza di E0.")

    base_results, (base_I_all, base_E_all) = I_multil_general_mono(
        el_layers, series_layers,
        E0_arr[0], phi, W_layers, A, d, t_layers,
        enance=enance, verbose=verbose
    )

    n_layer = len(base_results)
    template_E_layers = [base_results[k][1] for k in range(n_layer)]
    agg_counts_layers = [np.zeros_like(base_results[k][0], dtype=float) for k in range(n_layer)]

    # first incident energy
    for k in range(n_layer):
        I_k, _ = base_results[k]
        agg_counts_layers[k] += w_arr[0] * I_k

    # remaining incident energies
    for idx_E in range(1, n_E):
        E0_k = E0_arr[idx_E]
        w_k  = w_arr[idx_E]

        res_k, (I_all_k, E_all_k) = I_multil_general_mono(
            el_layers, series_layers,
            E0_k, phi, W_layers, A, d, t_layers,
            enance=enance, verbose=False
        )

        if not np.allclose(E_all_k, base_E_all, atol=atol_E, rtol=0):
            raise ValueError("Le energie delle righe cambiano tra diversi E0.")

        for k in range(n_layer):
            I_k, E_k = res_k[k]
            if not np.allclose(E_k, template_E_layers[k], atol=atol_E, rtol=0):
                raise ValueError(f"Mismatch energie righe nel layer {k}.")
            agg_counts_layers[k] += w_k * I_k

    results_per_layer = []
    I_all_list, E_all_list = [], []

    for k in range(n_layer):
        I_k_tot = agg_counts_layers[k]
        E_k     = template_E_layers[k]
        results_per_layer.append((I_k_tot, E_k))
        I_all_list.append(I_k_tot)
        E_all_list.append(E_k)

    I_all = np.concatenate(I_all_list) if I_all_list else np.array([])
    E_all = np.concatenate(E_all_list) if E_all_list else np.array([])

    if verbose:
        print("\n=== Spettro multi-energia ===")
        print(f"E0 list = {E0_arr}")
        print(f"weights = {w_arr}")

    return results_per_layer, (I_all, E_all)


# =========================
# Element labels for peaks in each layer
# =========================

def _labels_for_layer(el_layer, series_layer):
    labels = []
    for el, ser in zip(el_layer, series_layer):
        lines, _ = set_line(ser)
        for _ in lines:
            labels.append(el)
    return labels


def I_multil_to_2xN_with_symbols(
        el_layers, series_layers, E0, phi, W_layers, A, d, t_layers,
        enance=True,
        sum_same_energy=True
    ):
    """Convert multilayer line intensities into a 2-by-N [counts; energies] array and corresponding element labels."""
    res_layers, (I_all_flat, E_all_flat) = I_multil_general(
        el_layers, series_layers, E0, phi, W_layers, A, d, t_layers, enance=enance
    )

    labels_all = []
    I_all = []
    E_all = []

    for k, (I_k, E_k) in enumerate(res_layers):
        lab_k = _labels_for_layer(el_layers[k], series_layers[k])

        I_k = np.asarray(I_k, dtype=float)
        E_k = np.asarray(E_k, dtype=float)

        if len(lab_k) != len(I_k):
            raise ValueError(f"Layer {k}: mismatch label/righe.")

        labels_all.extend(lab_k)
        I_all.append(I_k)
        E_all.append(E_k)

    if not I_all:
        return np.zeros((2,0)), []

    I_all = np.concatenate(I_all)
    E_all = np.concatenate(E_all)
    labels_all = np.asarray(labels_all, dtype=object)

    mask = (E_all > 0.0) & (I_all > 0.0)
    I_all = I_all[mask]
    E_all = E_all[mask]
    labels_all = labels_all[mask]

    if I_all.size == 0:
        return np.zeros((2,0)), []

    if sum_same_energy:
        keys = np.core.records.fromarrays(
            [E_all, labels_all],
            names="E,sym"
        )
        uniq_keys, inv = np.unique(keys, return_inverse=True)
        I_sum = np.zeros(len(uniq_keys), dtype=float)
        for idx, val in enumerate(I_all):
            I_sum[inv[idx]] += val

        energies = uniq_keys["E"]
        peak_symbols = uniq_keys["sym"].tolist()
        counts = I_sum
    else:
        counts = I_all
        energies = E_all
        peak_symbols = labels_all.tolist()

    I_bll2 = np.vstack([counts, energies])
    return I_bll2, peak_symbols


# =========================
# Multilayer spectral components (mono- or polychromatic)
# =========================

def build_multilayer_component_spectra(
        el_layers, series_layers,
        E0,                 # scalar or array
        phi, W_layers, A, d, t_layers,
        enance=True,
        weights=None
    ):
    """Build total, intra-layer, and layer-resolved inter-layer spectral components for a mono- or polychromatic incident source."""

    if np.isscalar(E0):
        E0_list = np.array([float(E0)], dtype=float)
        if weights is None:
            weights = np.array([1.0], dtype=float)
        else:
            weights = np.array(weights, dtype=float).reshape(-1)
            if weights.size != 1:
                raise ValueError("Se E0 è scalare, weights deve avere un solo valore.")
    else:
        E0_list = np.array(E0, dtype=float).reshape(-1)
        if weights is None:
            weights = np.ones_like(E0_list, dtype=float)
        else:
            weights = np.array(weights, dtype=float).reshape(-1)
            if weights.size != E0_list.size:
                raise ValueError("weights deve avere la stessa lunghezza di E0.")

    n_layer = len(el_layers)

    total_counts_all, total_energy_all, total_syms_all = [], [], []
    intra_counts_all, intra_energy_all, intra_syms_all = [], [], []

    inter_layers_I = []
    inter_layers_E = []
    inter_layers_sym = []

    for k in range(n_layer):
        elements_k = el_layers[k]
        series_k   = series_layers[k]
        W_k        = W_layers[k]
        t_k        = t_layers[k]

        # reference pass
        E0_ref = E0_list[0]
        I_k_intra_ref, E_k_lines_ref = I_line_t(
            elements_k, series_k,
            E0_ref, phi, W_k, A, d, t_k,
            enance
        )
        I_k_intra_ref = np.asarray(I_k_intra_ref, dtype=float)
        E_k_lines_ref = np.asarray(E_k_lines_ref, dtype=float)

        n_lines_k = I_k_intra_ref.size
        labels_layer = _labels_for_layer(elements_k, series_k)

        I_k_intra_acc = np.zeros(n_lines_k, dtype=float)
        I_k_inter_acc = np.zeros(n_lines_k, dtype=float)
        I_k_total_acc = np.zeros(n_lines_k, dtype=float)

        for E0_val, w in zip(E0_list, weights):

            # cache emission lines and incident attenuation for this E0
            layer_lines_cache = []
            for j in range(n_layer):
                elements_j = el_layers[j]
                series_j   = series_layers[j]
                E_lines_j, meta_j_list = list_lines_for_layer(elements_j, series_j, E0_val)
                layer_lines_cache.append((E_lines_j, meta_j_list))

            mu_lambda_layers = [
                mu_eff(el_layers, W_layers, j, E0_val, True, phi) for j in range(n_layer)
            ]

            I_k_intra, E_k_lines = I_line_t(
                elements_k, series_k,
                E0_val, phi, W_k, A, d, t_k,
                enance
            )
            I_k_intra = np.asarray(I_k_intra, dtype=float)
            E_k_lines = np.asarray(E_k_lines, dtype=float)

            if not np.allclose(E_k_lines, E_k_lines_ref, rtol=0, atol=1e-6):
                raise ValueError(
                    f"Le energie delle righe nel layer {k} cambiano con E0."
                )

            sin_in, sin_out = np.sin(phi[0]), np.sin(phi[1])

            # incident attenuation is independent of emission-line energy
            mu_in_sum = 0.0
            for m in range(0, k):
                mu_in_sum  += _mu_layer_sum(el_layers[m], W_layers[m], E0_val,  kind='tot') * t_layers[m] / sin_in
            A_in = np.exp(-mu_in_sum)
            A_in_arr = np.full_like(E_k_lines, A_in, dtype=float)

            # outgoing attenuation depends on emission-line energy
            A_out_list = []
            for E_line in E_k_lines:
                mu_out_sum = 0.0
                for m in range(0, k):
                    mu_out_sum += _mu_layer_sum(el_layers[m], W_layers[m], E_line,  kind='tot') * t_layers[m] / sin_out
                A_out_list.append(np.exp(-mu_out_sum))

            A_out_arr = np.asarray(A_out_list)

            I_k_corr_intra = I_k_intra * A_in_arr * A_out_arr

            _, meta_k_list = layer_lines_cache[k]
            assert len(meta_k_list) == len(E_k_lines)

            S_tot_list = []
            I_k_inter  = []

            for idx, (I_i0, E_i, meta_i) in enumerate(
                    zip(I_k_corr_intra, E_k_lines, meta_k_list)
            ):
                S_tot, _ = S_inter_for_line(
                    k, idx, I_i0, E_i, meta_i,
                    el_layers, series_layers, W_layers, t_layers,
                    phi, E0_val,
                    layer_lines_cache=layer_lines_cache,
                    mu_lambda_layers=mu_lambda_layers
                )
                S_tot_list.append(S_tot)
                I_k_inter.append(I_i0 * S_tot)

            S_tot_arr_E = np.asarray(S_tot_list, dtype=float)
            I_k_inter   = np.asarray(I_k_inter,   dtype=float)
            I_k_total   = I_k_corr_intra * (1.0 + S_tot_arr_E)

            I_k_intra_acc += w * I_k_corr_intra
            I_k_inter_acc += w * I_k_inter
            I_k_total_acc += w * I_k_total

        total_counts_all.append(I_k_total_acc)
        total_energy_all.append(E_k_lines_ref)
        total_syms_all.extend(labels_layer)

        intra_counts_all.append(I_k_intra_acc)
        intra_energy_all.append(E_k_lines_ref)
        intra_syms_all.extend(labels_layer)

        inter_layers_I.append(I_k_inter_acc)
        inter_layers_E.append(E_k_lines_ref)
        inter_layers_sym.append(list(labels_layer))

    if total_counts_all:
        total_counts_all = np.concatenate(total_counts_all)
        total_energy_all = np.concatenate(total_energy_all)
        intra_counts_all = np.concatenate(intra_counts_all)
        intra_energy_all = np.concatenate(intra_energy_all)
    else:
        total_counts_all = np.array([])
        total_energy_all = np.array([])
        intra_counts_all = np.array([])
        intra_energy_all = np.array([])

    total_syms_all = list(total_syms_all)
    intra_syms_all = list(intra_syms_all)

    I_bll2_total = np.vstack([total_counts_all, total_energy_all])
    I_bll2_intra = np.vstack([intra_counts_all, intra_energy_all])

    inter_bll2_per_layer = []
    for I_k_inter, E_k, sym_k in zip(inter_layers_I, inter_layers_E, inter_layers_sym):
        inter_bll2_per_layer.append(
            (np.vstack([np.asarray(I_k_inter), np.asarray(E_k)]),
             list(sym_k))
        )

    return I_bll2_total, total_syms_all, I_bll2_intra, intra_syms_all, inter_bll2_per_layer


# =========================
# Plotting and TXT export
# =========================

def plot_multilayer_voigt_components(
        json_path: str,
        el_layers, series_layers,
        E0,                 # scalar or array-like
        phi, W_layers, A, d, t_layers,
        enance: bool = True,
        x_min_eV: float = 0.0,
        x_max_eV: float = 40000.0,
        x_points: int = 8000,
        axis_limits: list[float] | None = None,
        show_total: bool = True,
        show_intra: bool = False,
        show_inter_per_layer: bool = True,
        show_voigt_components: bool = False,
        add_noise_total: bool = True,
        noise_mode: str = "total",
        random_state: int | None = None,
        save: bool = False,
        title_file: str | None = None,
        weights=None,
        do_plot: bool = True
    ):

    (I_bll2_total, peak_syms_total,
     I_bll2_intra, peak_syms_intra,
     inter_bll2_per_layer) = build_multilayer_component_spectra(
         el_layers, series_layers,
         E0, phi, W_layers, A, d, t_layers,
         enance=enance,
         weights=weights
    )

    common = load_common_dict_raw(json_path)

    x = np.linspace(x_min_eV, x_max_eV, x_points)
    y_total_voigt = np.zeros_like(x)
    y_main_for_output = np.zeros_like(x)

    rng = np.random.default_rng(random_state)
    if do_plot:
        plt.figure(figsize=(10,5))

    def _accumulate_voigt(I_bll2, peak_syms, label_prefix="", alpha=1.0, lw=2.0, draw_G_L=False):
        intens   = I_bll2[0].astype(float)
        energies = I_bll2[1].astype(float)
        N = intens.size

        y_sum = np.zeros_like(x)

        for idx in range(N):
            Epk = float(energies[idx])
            Ipk = float(intens[idx])
            sym = peak_syms[idx] if peak_syms is not None else f"pk{idx+1}"

            # efficiency and natural FWHM from JSON
            eff, fwhm_L = _eff_and_width_from_common(common, sym, Epk)

            I_eff  = Ipk * eff
            fwhm_G = gaussian_fwhm_eV_from_keV(Epk/1000.0)

            # compute Gaussian and Lorentzian components only when plotting them
            if draw_G_L and do_plot:
                G = gaussian_profile(x, Epk, fwhm_G) * I_eff
                L = lorentz_profile (x, Epk, fwhm_L) * I_eff
                plt.plot(x, G, ls="--", alpha=0.4*alpha,
                         label=f"{label_prefix}{sym} G" if idx==0 else None)
                plt.plot(x, L, ls=":",  alpha=0.4*alpha,
                         label=f"{label_prefix}{sym} L" if idx==0 else None)

            # always calculate the Voigt profile
            V = voigt_profile(x, Epk, fwhm_G, fwhm_L) * I_eff
            y_sum += V

        if do_plot:
            plt.plot(x, y_sum, lw=lw, alpha=alpha,
                     label=label_prefix.strip())
        return y_sum

    # total spectrum
    if show_total and I_bll2_total.size > 0:
        y_total_voigt = _accumulate_voigt(
            I_bll2_total, peak_syms_total,
            label_prefix="Totale (intra+inter)",
            alpha=1.0,
            lw=2.0,
            draw_G_L=show_voigt_components
        )

        if add_noise_total:
            if noise_mode not in ("total",):
                raise ValueError("Per ora noise_mode supportato è solo 'total'.")
            y_noisy = rng.poisson(np.clip(y_total_voigt, 0, None))
            if do_plot:
                plt.plot(x, y_noisy, lw=1.5, alpha=0.9,
                         label="Totale + Poisson")
            y_main_for_output = y_noisy
        else:
            y_main_for_output = y_total_voigt

    # intra-layer contribution only
    if show_intra and I_bll2_intra.size > 0:
        _accumulate_voigt(
            I_bll2_intra, peak_syms_intra,
            label_prefix="Intra-layer only",
            alpha=0.7,
            lw=1.5,
            draw_G_L=False
        )

    # inter-layer contribution by layer
    if show_inter_per_layer:
        for k, (I_bll2_k, syms_k) in enumerate(inter_bll2_per_layer):
            if I_bll2_k.size == 0:
                continue
            _accumulate_voigt(
                I_bll2_k, syms_k,
                label_prefix=f"Inter-layer layer {k}",
                alpha=0.8,
                lw=1.5,
                draw_G_L=False
            )

    if do_plot:
        plt.xlabel("Energy (eV)")
        plt.ylabel("Counts (a.u.)")
        plt.title("Multilayer XRF: Voigt + componenti")

        if axis_limits is not None:
            plt.axis(axis_limits)
        else:
            plt.xlim(x_min_eV, x_max_eV)

        handles, labels = plt.gca().get_legend_handles_labels()
        labels_clean, handles_clean = [], []
        for h, lab in zip(handles, labels):
            if lab and lab not in labels_clean:
                labels_clean.append(lab)
                handles_clean.append(h)
        plt.legend(handles_clean, labels_clean, fontsize=8, ncol=2, loc="best")

        plt.tight_layout()
        plt.show()

    # TXT export
    spectrum_array = np.vstack([x, y_main_for_output])

    if save and title_file:
        # CREATE THE DIRECTORY IF NEEDED
        out_dir = os.path.dirname(title_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        data_to_save = np.column_stack([x, y_main_for_output])
        np.savetxt(title_file,
                   data_to_save,
                   fmt="%.6e",
                   header="Energy_eV   Counts",
                   comments="")
        print(f"File salvato: {title_file}")

    return spectrum_array


# In[1]:


import json
import gzip

def pigment_code(name: str) -> str:
    """Convert a pigment name into a short identifier for spectrum filenames."""
    clean = name.replace("_", " ").replace("-", " ")
    words = [w for w in clean.split() if w]
    if len(words) == 1:
        return (words[0][:2]).upper()
    else:
        return "".join(w[0].upper() for w in words)

def save_simulation_to_json(
        filename: str,
        spectrum_array,
        painting,
        max_layers: int = 3      # <-- L_max = 3 (2 paint + support)
    ):
    """Serialize a simulated spectrum and its stratigraphy using three fixed logical slots: surface paint, optional intermediate paint, and ground."""

    if max_layers != 3:
        print(f"[WARN] save_simulation_to_json pensata per max_layers=3, ricevuto {max_layers}")

    E_axis_eV = spectrum_array[0].astype(float).tolist()
    counts    = spectrum_array[1].astype(float).tolist()

    layers = painting
    n_layers = len(layers)

    # the final physical layer is the ground; preceding layers are paint
    if n_layers < 2:
        raise ValueError(f"Painting con meno di 2 layer (paint + support): n_layers={n_layers}")

    support_layer = layers[-1]
    paint_layers  = layers[:-1]
    n_paint       = len(paint_layers)

    if n_paint not in (1, 2):
        print(f"[WARN] Numero di layer pittorici inaspettato: {n_paint}. Gestisco comunque.")

    # initialize slots to zero
    layer_present = [0, 0, 0]
    rho_t_slots   = [0.0, 0.0, 0.0]
    thickness_um_comment = [0.0, 0.0, 0.0]
    rho_slots     = [0.0, 0.0, 0.0]
    organic_frac  = [0.0, 0.0, 0.0]   # store H+C+O+N in each slot
    composition   = {}

    ORGANIC_SET = {"H", "C", "O", "N"}

    def fill_slot(slot_idx: int, layer_dict):
        """Populate a zero-based output slot with the composition and metadata of a physical layer."""
        layer_present[slot_idx] = 1

        rho_layer = float(layer_dict["rho_layer_g_cm3"])
        thick_um  = float(layer_dict["thickness_um"])

        rho_slots[slot_idx] = rho_layer
        thickness_um_comment[slot_idx] = thick_um

        # ρ * t in cm (t in μm → cm con 1e-4)
        rho_t_slots[slot_idx] = rho_layer * thick_um * 1e-4

        # separate light organic elements (H,C,O,N) from inorganic elements
        elem_comp = layer_dict.get("elemental_composition", {})

        comp_inorg = {}
        org_sum = 0.0
        for el, frac in elem_comp.items():
            f = float(frac)
            if el in ORGANIC_SET:
                org_sum += f
            else:
                comp_inorg[el] = f

        organic_frac[slot_idx] = org_sum
        composition[f"layer_{slot_idx+1}"] = comp_inorg

    # --- Slot 1: always the surface paint layer ---
    fill_slot(0, paint_layers[0])

    # --- Slot 2: intermediate paint layer if present ---
    if n_paint >= 2:
        fill_slot(1, paint_layers[1])
    # otherwise leave the slot empty

    # --- Slot 3: always the ground layer ---
    fill_slot(2, support_layer)

    # --- spectrum_id with the actual pigments used (paint[1..n_paint] + support) ---
    used_layers = list(paint_layers) + [support_layer]
    pigment_codes = [pigment_code(layer["pigment_name"]) for layer in used_layers]
    spectrum_id = "sim_" + "_".join(pigment_codes)

    data = {
        "spectrum_id": spectrum_id,
        "E_axis_eV": E_axis_eV,
        "counts": counts,
        "labels": {
            "layer_present": layer_present,
            "thickness_rho_t": rho_t_slots,
            "thickness_um_comment": thickness_um_comment,
            "organic_frac": organic_frac,   # <-- H+C+O+N for each layer
            "rho": rho_slots,
            "composition": composition      # <-- inorganic elements only
        },
        "sim_meta": {"F_true": 1.0}
    }

    import json
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Salvato file: {filename}")



def save_simulation_to_json1(
        filename: str,
        spectrum_array,
        painting,
        max_layers: int = 5
    ):
    """Save the simulation to an uncompressed JSON file using the original data structure."""
    E_axis_eV = spectrum_array[0].astype(float).tolist()
    counts    = spectrum_array[1].astype(float).tolist()

    layers = painting
    n_layers = len(layers)

    if n_layers > max_layers:
        print(f"ATTENZIONE: {n_layers} layer, ma max_layers={max_layers}. Taglio ai primi {max_layers}.")
        n_layers = max_layers

    layer_present = [1]*n_layers + [0]*(max_layers - n_layers)

    rho_t = [
        float(layers[i]["rho_layer_g_cm3"]) * float(layers[i]["thickness_um"]) * 1e-4
        for i in range(n_layers)
    ] + [0.0]*(max_layers - n_layers)

    thickness_um_comment = [
        float(layers[i]["thickness_um"]) for i in range(n_layers)
    ] + [0.0]*(max_layers - n_layers)

    rho = [
        float(layers[i]["rho_layer_g_cm3"])
        for i in range(n_layers)
    ] + [0.0]*(max_layers - n_layers)

    organic_frac = [0.0]*max_layers

    composition = {}
    for i in range(n_layers):
        elem_comp = layers[i].get("elemental_composition", {})
        comp_clean = {el: float(frac) for el, frac in elem_comp.items()}
        composition[f"layer_{i+1}"] = comp_clean

    pigment_codes = [pigment_code(layers[i]["pigment_name"]) for i in range(n_layers)]
    spectrum_id = "sim_" + "_".join(pigment_codes)

    data = {
        "spectrum_id": spectrum_id,
        "E_axis_eV": E_axis_eV,
        "counts": counts,
        "labels": {
            "layer_present": layer_present,
            "thickness_rho_t": rho_t,
            "thickness_um_comment": thickness_um_comment,
            "organic_frac": organic_frac,
            "rho":rho,
            "composition": composition
        },
        "sim_meta": {"F_true": 1.0}
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Salvato file: {filename}")


def save_simulation_to_json_gz(
        filename: str,
        spectrum_array,
        painting,
        max_layers: int = 5
    ):
    """Save the same simulation data structure to a gzip-compressed JSON file."""
    E_axis_eV = spectrum_array[0].astype(float).tolist()
    counts    = spectrum_array[1].astype(float).tolist()

    layers = painting
    n_layers = len(layers)

    if n_layers > max_layers:
        print(f"ATTENZIONE: {n_layers} layer, ma max_layers={max_layers}. Taglio ai primi {max_layers}.")
        n_layers = max_layers

    layer_present = [1]*n_layers + [0]*(max_layers - n_layers)

    rho_t = [
        float(layers[i]["rho_layer_g_cm3"]) * float(layers[i]["thickness_um"]) * 1e-4
        for i in range(n_layers)
    ] + [0.0]*(max_layers - n_layers)

    thickness_um_comment = [
        float(layers[i]["thickness_um"]) for i in range(n_layers)
    ] + [0.0]*(max_layers - n_layers)

    rho = [
        float(layers[i]["rho_layer_g_cm3"])
        for i in range(n_layers)
    ] + [0.0]*(max_layers - n_layers)

    organic_frac = [0.0]*max_layers

    composition = {}
    for i in range(n_layers):
        elem_comp = layers[i].get("elemental_composition", {})
        comp_clean = {el: float(frac) for el, frac in elem_comp.items()}
        composition[f"layer_{i+1}"] = comp_clean

    pigment_codes = [pigment_code(layers[i]["pigment_name"]) for i in range(n_layers)]
    spectrum_id = "sim_" + "_".join(pigment_codes)

    data = {
        "spectrum_id": spectrum_id,
        "E_axis_eV": E_axis_eV,
        "counts": counts,
        "labels": {
            "layer_present": layer_present,
            "thickness_rho_t": rho_t,
            "thickness_um_comment": thickness_um_comment,
            "organic_frac": organic_frac,
            "composition": composition
        },
        "sim_meta": {"F_true": 1.0}
    }

    with gzip.open(filename, "wt", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Salvato file compresso: {filename}")


# In[14]:


import os
import numpy as np
# ... additional imports

# ---------------------------------------------------------
# Worker: generate and save a JSON spectrum from a painting
# ---------------------------------------------------------

def generate_spectrum_and_save(
    idx: int,
    painting,
    json_path: str,
    out_dir: str,
    E0,
    phi,
    A,
    d_geom,
    weights,
    x_min_eV: float = 0.0,
    x_max_eV: float = 40000.0,
    x_points: int = 2048,
    max_layers: int = 9,
    do_plot: bool = False,
    add_noise_total: bool = False,
):
    """Generate and save one simulated spectrum from a paint stratigraphy. Intended for parallel dataset generation."""

    os.makedirs(out_dir, exist_ok=True)

    # 1) build multilayer model inputs from the painting
    el_layers, series_layers, rho, t_l, W_ml = build_spectrum_input_from_painting(
        painting,
        include_support=True
    )
    t_layers=rho*t_l
    # 2) compute the spectrum
    spectrum_array = plot_multilayer_voigt_components(
        json_path=json_path,
        el_layers=el_layers,
        series_layers=series_layers,
        E0=E0,
        phi=phi,
        W_layers=W_ml,
        A=A,
        d=d_geom,
        t_layers=t_layers,
        enance=True,
        x_min_eV=x_min_eV,
        x_max_eV=x_max_eV,
        x_points=x_points,
        show_total=True,
        show_intra=False,
        show_inter_per_layer=False,
        show_voigt_components=False,
        add_noise_total=add_noise_total,
        noise_mode="total",
        random_state=None,
        save=False,
        title_file=None,
        weights=weights,
        do_plot=do_plot
    )

    counts=spectrum_array[1].astype(float)

    if np.isnan(counts).any() or np.isinf(counts).any():
        print(f"[Warn] Spettri {idx} contiene NaN/Inf")
        return idx, None
    else:
        # 3) save JSON output
        file_out = os.path.join(out_dir, f"sim_{idx:05d}.json")
        save_simulation_to_json(
            file_out,
            spectrum_array,
            painting,
            max_layers=max_layers
        )
    
        return idx, file_out


# In[2]:


def simulate_xrf_spectrum_numpy(
    json_path,
    el_layers,
    series_layers,
    W_layers,
    t_layers,
    phi,
    A,
    d,
    E0,
    weights,
    F_global,
    B0,
    B1,
    x_min_eV,
    x_max_eV,
    x_points,
):
    """Run the physical XRF simulation and reconstruct a continuous spectrum on a NumPy energy axis."""

    # 1. COMPONENTS (intra- and inter-layer)
    (
        I_bll2_total, 
        peak_syms_total,
        _,
        _,
        _,
    ) = build_multilayer_component_spectra(
            el_layers=el_layers,
            series_layers=series_layers,
            E0=E0,
            phi=phi,
            W_layers=W_layers,
            A=A,
            d=d,
            t_layers=t_layers,
            enance=True,
            weights=weights
        )

    # 2. LOAD common_dict (efficiencies and natural linewidths)
    common = load_common_dict_raw(json_path)

    # 3. Energy axis
    x = np.linspace(x_min_eV, x_max_eV, x_points)
    y = np.zeros_like(x)

    # 4. Voigt line-shape reconstruction
    intens = I_bll2_total[0]
    energies = I_bll2_total[1]

    for Ipk, Epk, sym in zip(intens, energies, peak_syms_total):

        # efficiency and natural FWHM
        eff, fwhm_L = _eff_and_width_from_common(common, sym, float(Epk))
        I_eff = Ipk * eff * F_global

        # instrumental Gaussian FWHM
        fwhm_G = gaussian_fwhm_eV_from_keV(Epk / 1000.0)

        # Voigt profile
        V = voigt_profile(x, Epk, fwhm_G, fwhm_L) * I_eff
        y += V

    # 5. Linear background
    y += B0 + B1 * (x - x_min_eV)

    return x, y


# In[ ]:




