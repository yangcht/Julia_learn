module Jadex
### Sections
# * Constants
# * Data file parser
# * Run descriptor
# * Escape probability
# * Background
# * Matrix
# * Grid calculator
###

import Base: show, showcompact


##############################################################################
# Constants
##############################################################################
# Physical constants in CGS
const clight  = 2.99792458e10   # speed of light     (cm/s)
const hplanck = 6.6260963e-27   # Planck constant    (erg/Hz)
const kboltz  = 1.3806505e-16   # Boltzmann constant (erg/K)
const amu     = 1.67262171e-24  # atomic mass unit   (g)
const fk      = hplanck * clight / kboltz
const thc     = 2 * hplanck * clight

# Mathematical constants
const fgauss  = √π / (2 * √log(2)) * 8π

const miniter = 10     # minimum number of iterations
const maxiter = 9999   # maximum number of iterations

const ccrit   = 1e-6   # relative tolerance of solution
const eps     = 1e-30  # round-off error
const minpop  = 1e-20  # minimum level population

# Directories
const datadir = "../data/"


##############################################################################
# Data file parser
##############################################################################
# Parse molecular data files and create input types.

immutable CollisionPartner
    collref::String  # collision partner reference name
    ncoll::Int  # number of collisional transitions
    ntemp::Int  # number of collisional temperatures
    temp::Vector{Float64}  # temperatures
    lcu::Vector{Float64}  # upper state of collision
    lcl::Vector{Float64}  # lower state of collision
    coll::Matrix{Float64}  # collision rates, cm^3 s^-1
end
const valid_partners = ["h2", "p-h2", "o-h2", "e", "h", "he", "h+"]


immutable Molecule
    # Header
    specref::String  # molecule
    amass::Int  # molecular weight
    # Energy levels
    nlev::Int  # number of energy levels
    eterm::Vector{Float64}  # energy levels, in cm^-1
    gstat::Vector{Float64}  # statistical weights
    qnum::Vector{ASCIIString}  # quantum numbers
    # Transitions
    nline::Int  # number of radiative transitions
    iupp::Vector{Int}  # upper state
    ilow::Vector{Int}  # lower state
    aeinst::Vector{Float64}  # Einstein A
    spfreq::Vector{Float64}  # spectral line frequencies
    eup::Vector{Float64}  # upper state energy, E_u / k
    xnu::Vector{Float64}  # difference in energy levels between up/low
    # Collision rates
    npart::Int  # number of collision partners
    colliders::Vector{CollisionPartner}  # list of colliders
end
function Molecule(specref::String)
    # Read in data file
    f = datadir * specref * ".dat" |> open |> readlines
    f = [strip(l) for l in f]
    # Parse header
    amass = f[4] |> float
    # Parse energies
    nlev = f[6] |> int
    eterm = Array(Float64, nlev)
    gstat = Array(Float64, nlev)
    qnum = Array(String, nlev)
    for (ii,jj) in enumerate(8:7+nlev)
        l = f[jj] |> split
        eterm[ii] = l[2] |> float
        gstat[ii] = l[3] |> float
        qnum[ii] = l[4]
    end
    # Parse transitions
    nline = f[9+nlev] |> int
    iupp = Array(Int, nline)
    ilow = Array(Int, nline)
    aeinst = Array(Float64, nline)
    spfreq = Array(Float64, nline)
    eup = Array(Float64, nline)
    xnu = Array(Float64, nline)
    for (ii,jj) in enumerate(11+nlev:10+nlev+nline)
        l = f[jj] |> split
        iupp[ii] = l[2] |> int
        ilow[ii] = l[3] |> int
        aeinst[ii] = l[4] |> float
        spfreq[ii] = l[5] |> float
        eup[ii] = l[6] |> float
        xnu[ii] = eterm[iupp[ii]] - eterm[ilow[ii]]
    end
    # Parse collision partners
    npart = f[12+nlev+nline] |> int
    icolliders = 0
    colliders = Array(CollisionPartner, npart)
    for (ii,line) in enumerate(f)
        if line == "!COLLISIONS BETWEEN"
            icolliders += 1
            collref = valid_partners[int(f[ii+1][1:1])]
            ncoll = f[ii+3] |> int
            ntemp = f[ii+5] |> int
            temp = f[ii+7] |> split |> float
            lcu = Array(Int, ncoll)
            lcl = Array(Int, ncoll)
            coll = Array(Float64, ncoll, ntemp)
            for (jj,kk) in enumerate(9+ii:8+ii+ncoll)
                row = split(f[kk])
                lcu[jj] = row[2] |> int
                lcl[jj] = row[3] |> int
                coll[jj,:] = float(row[4:end])
            end
            colliders[icolliders] = CollisionPartner(collref, ncoll, ntemp, temp, lcu, lcl, coll)
        end
    end
    # TODO throw exception if no colliders
    if length(colliders) == 0
        warn("No colliders found.")
        throw(DomainError())
    end
    Molecule(specref, amass, nlev, eterm, gstat, qnum, nline, iupp, ilow,
        aeinst, spfreq, eup, xnu, npart, colliders)
end


function show(io::IO, mol::Molecule)
    collnames = join([c.collref for c in mol.colliders], ",")
    print("Molecule(specref=$(mol.specref), nlev=$(mol.nlev), " *
          "nline=$(mol.nline), colliders=$(collnames))")
end


function show(io::IO, col::CollisionPartner)
    print("CollisionPartner(collref=$(col.collref), ncoll=$(col.ncoll), " *
          "ntemp=$(col.ntemp))")
end


##############################################################################
# Run descriptor
##############################################################################
# Top level container to describe a model calculation.

immutable RunDef
    mol::Molecule  # molecule container
    collref::String  # name of collision partner to use
    density::Vector{Float64}  # number densities of collision partners, cm^-3
    totdens::Float64  # total number density of all partners, cm^-3
    freq::(Float64, Float64)  # lower and upper frequency boundaries, GHz
    tkin::Float64  # kinetic temperature, K
    cdmol::Float64  # molecular column density, cm^-2
    deltav::Float64  # FWHM line width, cm s^-1
    escprob::Function  # escape probability geometry
    bg::Background  # radiation field background
end


##############################################################################
# Escape probability
##############################################################################
# Functions to compute the escape probability

function βsphere(τ::Real)
    τr = τ / 2.0
    # Uniform sphere formula from Osterbrock (Astrophysics of Gaseous Nebulae
    # and Active Galactic Nuclei) Appendix 2 with power law approximations for
    # large and small tau
    abs(τr) < 0.1 ?
        1.0 - 0.75 * τr + τr^2 / 2.5 - τr^3 / 6.0 + τr^4 / 17.5 :
    abs(τr) > 50 ?
        0.75 / τr :
        0.75 / τr * (1.0 - 1.0 / 2τr^2) + (1.0 / τr + 1.0 / 2τr^2) * exp(-τ)
end


function βlvg(τ::Real)
    τr = τ / 2.0
    # Expanding sphere, Large Velocity Gradient, or Sobolev case. Formular from
    # De Jong, Boland, and Dalgarno (1980, A&A 91, 68).
    # Corrected by factor of 2 in order to return 1 for tau=1.
    abs(τr) < 0.01 ?
        1.0 :
    abs(τr) < 7.0 ?
        2.0 * (1.0 - exp(-2.34τr)) / 4.65τr :
        4τr * √log(τr / √π) \ 2.0
end


function βslab(τ::Real)
    # Slab geometry (e.g. shocks): de Jong, Dalgarno, and Chu (1975), ApJ 199,
    # 69 (again with power law approximations)
    abs(3τ) < 0.1  ?
        1.0 - 1.5 * (τ + τ^2) :
    abs(3τ) > 50.0 ?
        1.0 / 3τ :
        (1.0 - exp(-3τ)) / 3τ
end


##############################################################################
# Background
##############################################################################
# Compute the background radiation field

immutable Background
    trj::Vector{Float64}  # radiation temperatures
    backi::Vector{Float64}  # background intensity
    totalb::Vector{Float64}  # total flux
end


function bb(xnu::Array, tbg::Real=2.725)
    nline = length(xnu)
    trj = Array(Float64, nline)
    backi = Array(Float64, nline)
    totalb = Array(Float64, nline)
    for iline = 1:nline
        hnu = fk * xnu[iline] / tbg
        if hnu >= 160.0
            backi[iline] = eps
        else
            backi[iline] = thc * xnu[iline]^3 / (exp(fk * xnu[iline] / tbg) - 1.0)
        end
    end
    trj[:] = tbg
    totalb[:] = backi[:]
    Background(trj, backi, totalb)
end


##############################################################################
# Matrix
##############################################################################
# Compute the level populations

type Solution
    rhs::Vector{Float64}
    yrate::Matrix{Float64}
    # TODO add further quantities
end
function Solution(rdef::RunDef)
    mol = rdef.mol
    nlev = mol.nlev
    # Initialize rate matrix
    rhs = zeros(nlev+1)
    yrate = zeros(nlev+1, nlev+1)
    for ilev = 1:nlev
        for jlev = 1:nlev
            yrate[ilev,jlev] = -eps * rdef.totdens
        end
        yrate[nlev+1, ilev] = 1.0
        rhs[ilev] = eps * rdef.totdens
        yrate[ilev,nlev+1] = eps * rdef.totdens
    end
    rhs[nlev+1] = eps * rdef.totdens
    # TODO Process and interpolate molecule data
    # Initialized solution container
    Solution(rhs, yrate)
end


# Contribution of radiative processed to the rate matrix. Modifies the solution in place.
function rad_proc!(sol::Solution, rdef::RunDef, niter)
    # TODO need to access correct attributes of container types, right now
    # they are just referred to and will get access errors
    # First iteration, use background intensity
    if niter == 0
        for ii = 1:mol.nline
            mm = mol.iupp[ii]
            nn = mol.ilow[ii]
            etr = fk * mol.xnu[ii] / rdf.bg.trj[ii]
            exr = etr >= 160.0 ? 0.0 : 1.0 / (exp(etr) - 1.0)
            sol.yrate[mm,mm] += mol.aeinst[ii] * (1.0 + exr)
            sol.yrate[nn,nn] += mol.aeinst[ii] * gstat[mm] * exr / gstat[nn]
            sol.yrate[mm,nn] -= mol.aeinst[ii] * (gstat[mm] / gstat[nn]) * exr
            sol.yrate[nn,mm] -= mol.aeinst[ii] * (1.0 + exr)
        end
    else
        # Subsequent iterations: use escape probability
        cddv = rdef.cdmol / rdef.deltav
        # Count optically thick lines
        nthick = 0
        nfat = 0
        for iline = 1:nline
            xt = mol.xnu[iline]^3.0
            m  = mol.iupp[iline]
            n  = mol.ilow[iline]
            # Calculate source fn
            hnu = fk * xnu[iline] / tex[iline]
            if hnu >= 160.0
                bnutex = 0.0
            else
                bnutex = thc * xt / (exp(fk * xnu[iline] / tex[iline]) - 1.0)
            end
            # Calculate line optical depth
            taul[iline] = cddv * (xpop[n] * gstat[m] / gstat[n] - xpop[m]) / (fgaus * xt / aeinst[iline])
            if taul[iline] > 1e-2; nthick += 1 end
            if taul[iline] > 1e5;  nfat   += 1 end
            # Use escape probability approx for internal intensity
            β   = escprob(taul[iline])
            bnu = totalb[iline] * β
            exr = bnu / (thc * xt)
            # Radiative contribution to the rate matrix
            yrate[m,m] = yrate[m,m] + aeinst[iline] * (β + exr)
            yrate[n,n] = yrate[n,n] + aeinst[iline] * (gstat[m] * exr / gstat[n])
            yrate[m,n] = yrate[m,n] - aeinst[iline] * (gstat[m] / gstat[n] * exr)
            yrate[n,m] = yrate[n,m] - aeinst[iline] * (β + exr)
        end
        # Warn user if convergence problems expected
        if niter == 1 && nfat > 0
            warn("Some lines have very high optical depth")
        end
    end
end


# Contribution for collisional processes to the rate matrix
function col_proc!(sol, mol)
    nlev = mol.nlev
    for ii = 1:nlev
        sol.yrate[ii,ii] = sol.yrate[ii,ii] + mol.ctot[ii]
        for jj = 1:nlev
            if ii != jj
                sol.yrate[ii,jj] -= mol.crate[jj,ii]
            end
        end
    end
end


# Level populations are the normalized RHS components
function pop_proc!(sol)
    total = sum(sol.rhs)
    sol.xpopold = copy(sol.xpop)
    # Limit population to minpop
    for ii = 1:nlev
        sol.xpop[ii] = max(minpop, sol.rhs[ii] / total)
    end
    # if first iteration, no old population
    if niter == 0; sol.xpopold = copy(sol.xpop) end
end


function rates(sol::Solution, rdef::RunDef, niter::Int, conv::Bool)
    mol = rdef.mol
    nlev = mol.nlev
    nline = mol.nline

    # Contribution of radiative processed to the rate matrix
    rad_proc!(sol, rdef, niter)
    # Contribution for collisional processes to the rate matrix
    col_proc!(sol, mol)
    # Invert the rate matrix `yrate`
    # TODO ccall on yrate
    # Level populations are the normalized RHS components
    pop_proc!(sol, niter)
    # Compute excitation temperatures of the lines
    tsum = 0.0
    for ii = 1:nline
        mm = iupp[ii]
        nn = ilow[ii]
        xt = xnu[ii]^3
        if niter == 0
            if xpop[nn] <= minpop || xpop[mm] <= minpop
                tex[ii] = totalb[ii]
            else
                tex[ii] = fk * xnu[ii] / (log(xpop[nn] * gstat[mm] / (xpop[mm] * gstat[nn])))
            end
        else
            if xpop[nn] <= minpop || xpop[mm] <= minpop
                itex = tex[ii]
            else
                itex = fk * xnu[ii]
            end
            # Only thick lines count for convergence
            if taul[ii] > 0.01
                tsum += abs((itex - tex[ii]) / itex)
            end
            # Update excitation temperature and optical depth
            tex[ii] = 0.5 * (itex + tex[ii])
            taul[ii] = cddv * (xpop[nn] * gstat[mm] / gstat[nn] - xpop[mm]) / (fgauss * xt / aeinst[ii])
        end
    end

    # Introduce a minimum number of iterations
    if niter >= miniter
        if nthick == 0
            conv = true
        elseif tsum / nthick < ccrit
            conv = true
        end
    end

    # Now do the underrelaxation
    for ii = 1:nlev
        xpop[ii] = 0.3 * xpop[ii] + 0.7 * xpopold[ii]
    end
end


function solve(rdef::RunDef)
    sol = Solution(rdef)
    for niter = 0:maxiter
        rates!(sol, rdef, niter, conv)
        if conv
            println("Finished in $niter iterations.")
            break
        end
    end
    if ~conv
        warn("Calculations did not converge in $maxiter iterations.")
    end
end


##############################################################################
# Grid calculator
##############################################################################
# Create grids of `RunDef`s and add definitions to relevant functions to
# accept grid input.
# TODO parallelize grid calculations


end  # module
