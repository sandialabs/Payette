! The MIT License

! Copyright (c) 2011 Tim Fuller

! License for the specific language governing rights and limitations under
! Permission is hereby granted, free of charge, to any person obtaining a
! copy of this software and associated documentation files (the "Software"),
! to deal in the Software without restriction, including without limitation
! the rights to use, copy, modify, merge, publish, distribute, sublicense,
! and/or sell copies of the Software, and to permit persons to whom the
! Software is furnished to do so, subject to the following conditions:

! The above copyright notice and this permission notice shall be included
! in all copies or substantial portions of the Software.

! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
! OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
! THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
! FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
! DEALINGS IN THE SOFTWARE.


!*****************************************************************************!

subroutine futalafu_chk(ui)
  !***************************************************************************!
  !
  ! Checks validity of user inputs
  ! Sets defaults for unspecified user input.
  ! Adjusts user input to be self-consistent.
  !
  !***************************************************************************!
  use futalafu_constants
  implicit none
  !..................................................................parameters
  !......................................................................passed
  real(kind=dk), dimension (*) :: ui
  !.......................................................................local
  logical :: eosmod
  character*12 iam
  parameter(iam='futalafu_chk' )
  !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ futalafu_chk
  ! pass parameters to local variables
  eosmod = ui(ipeosid) /= zero

  ! check elastic moduli, calculate defaults if necessary
  if(ui(ipk) <= zero) then
     if(ui(ipcs) <= zero .and. ui(ipr0) <= zero) then
        call faterr(iam, "Bulk modulus K must be positive")
     else
        call logmes("#---- setting mu based on rho and cs")
        ui(ipk) = ui(ipr0) * ui(ipcs) * ui(ipcs)
     end if
  end if
  if(ui(ipmu) <= zero) then
     if(ui(ipnu) .ne. zero) then
        ! Check if user specified NU but not G0
        call logmes("#---- setting g0 based on nu and b0")
        ui(ipmu) = three * (one - two * ui(ipnu)) / &
                  (two * (one + ui(ipnu))) * ui(ipk)
     else
        call faterr(iam, "Shear modulus MU must be positive")
     end if
  end if

  ! check strength parameters
  if(ui(ipy0) <= zero) call faterr(iam, "Yield strength Y must be positive")

  if(ui(ipa) < zero) &
       call faterr(iam, "Kinematic hardening modulus A must be non-negative")
  if(ui(ipc0) < zero) &
       call faterr(iam, "Isotropic hardening modulus C must be non-negative")
  if(ui(ipc1) < zero) then
     call faterr(iam, "Isotropic hardening power M must be non-negative")
  else if(ui(ipc1) == zero) then
     if(ui(ipc0) /= zero) call logmes("#---- C being set to 0 because M = 0")
     ui(ipc0) = zero
     ui(ipc1) = one
  end if
  if(ui(ipc2) < zero) &
       call faterr(iam, "Yield surface pressure term B must be non-negative")

  ! check damage parameters
  if(ui(ipyf) == zero) then
     ui(ipyf) = ui(ipy0)
  else if(ui(ipyf) < zero) then
     call faterr(iam, "Failed strength YF must be non-negative")
  end if
  if(ui(ipsf) <= zero) ui(ipsf) = ten
  if(ui(ipgf) <= zero) ui(ipgf) = huge(one)

  ! poisson's ratio
  ui(ipnu) = (three * ui(ipk) - two * ui(ipmu)) / (six * ui(ipk) + two * ui(ipmu))
  if(ui(ipnu) < zero) call logmes("#---- WARNING: negative Poisson's ratio")

  ! check the equation of state and energetic properties
  if(ui(ipt0) <= zero) ui(ipt0) = 298._dk
  if(eosmod) call check_eos(ui)

  return
end subroutine futalafu_chk


!*****************************************************************************!


subroutine futalafu_rxv(ui, nx, namea, keya, rinit, iadvct)
  !***************************************************************************!
  ! Request extra variables and set defaults
  !
  ! This subroutine creates lists of the internal state variables needed for
  ! futalafu. This routine merely sends a LIST of internal state variable
  ! requirements back to the host code. IT IS THE RESPONSIBILITY OF THE HOST
  ! CODE to loop over the items in each list to actually establish necessary
  ! storage and (if desired) set up plotting, restart, and advection control
  ! for each internal state variable.
  !
  ! called by: host code after all input data have been checked
  !***************************************************************************!
  use futalafu_constants
  implicit none

  !..................................................................parameters
  integer(kind=ik), parameter :: mmcn=50, mmck=10, mnunit=7
  integer(kind=ik), parameter :: mmcna=nxtra*mmcn, mmcka=nxtra*mmck
  !......................................................................passed
  integer(kind=ik) :: nx
  integer(kind=ik), dimension(*) :: iadvct
  real(kind=dk), dimension(*) :: ui, rinit
  character*1 namea(*), keya(*)

  !.......................................................................local
  integer(kind=ik) :: ij, i
  character*(mmcn) name(nxtra)
  character*(mmck) key(nxtra)
  character(1) :: char
  character(2), dimension(6) :: symmap=(/"11", "22", "33", "12", "23", "13"/)
  !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ futalafu_rxv

  call logmes('#---- requesting futalafu extra variables')

  rinit(1:nxtra) = zero

  nx = 0

  ! distortional plastic strain
  nx = nx + 1
  name(nx) = 'distortional plastic strain'
  key(nx) = 'GAM'
  iadvct(nx) = 0        ! input and output

  ! plastic volume strain
  nx = nx + 1
  name(nx) = 'plastic volume strain'
  key(nx) = 'EPV'
  iadvct(nx) = 0        ! input and output

  ! damage
  nx = nx + 1
  name(nx) = 'damage'
  key(nx) = 'DAM'
  iadvct(nx) = 0        ! input and output

  ! -- back stress
  do ij = 1, 6
     nx = nx + 1
     name(nx) = symmap(ij)//' component of back stress'
     key(nx) = 'BSIG'//symmap(ij)
     iadvct(nx) = 1
  end do

  ! specific energy
  nx = nx + 1
  name(nx) = 'Specific energy'
  key(nx) = 'ENRGY'
  iadvct(nx) = 1
  rinit(nx) = ui(ipcv) * ui(ipt0)

  ! temperature
  nx = nx + 1
  name(nx) = 'Temperature'
  key(nx) = 'TMPR'
  iadvct(nx) = 1
  rinit(nx) = ui(ipt0)

  ! density
  nx = nx + 1
  name(nx) = 'Density'
  key(nx) = 'RHO'
  iadvct(nx) = 1
  rinit(nx) = ui(ipr0)

  ! magnitude of deviatoric stress
  nx = nx + 1
  name(nx) = "Magnitude of deviatoric stress"
  key(nx) = "R"
  iadvct(nx) = 1

  ! magnitude of hydrostatic stress
  nx = nx + 1
  name(nx) = "Magnitude of hydrostatic stress"
  key(nx) = "Z"
  iadvct(nx) = 1

  ! yield surface flag
  nx = nx + 1
  name(nx) = "Yield flag"
  key(nx) = "YLD"
  iadvct(nx) = 1

  free: do i = 1, 5
     nx = nx + 1
     write(char, "(I1)") i
     name(nx) = "Free variable "//char
     key(nx) = "FREE0"//char
     iadvct(nx) = 1
  end do free

  call tokens(nx, name, namea)
  call tokens(nx, key, keya)
  return
end subroutine futalafu_rxv


!*****************************************************************************!


subroutine futalafu_calc(nc, nx, dt, ui, stressarg, d, xtra)
  !***************************************************************************!
  !
  ! Combined kinematic/isotropic hardening pressure dependent plasticity
  !
  !***************************************************************************!
  !
  ! input arguments
  ! ===============
  ! nc         int                   Number of blocks to be processed
  ! nx         int                   Number of internal state vars
  ! dt         dp                    Current time increment
  ! ui         dp,ar(nprop)          User inputs
  ! d          dp,ar(6)              Strain increment
  !
  ! input output arguments
  ! ======================
  ! stressarg dp,ar(6)                stress
  ! xtra      dp,ar(nx)               state variables
  !
  ! output arguments
  ! ================
  ! usm      dp                      uniaxial strain modulus
  !
  !***************************************************************************!
  !
  ! stresss and strains, plastic strain tensors
  ! 11, 22, 33, 12, 23, 13
  !
  !***************************************************************************!
  use tensor_toolkit
  use futalafu_constants
  implicit none
  !......................................................................passed
  integer(kind=ik), intent(in) :: nc, nx
  real(kind=dk), intent(in) :: dt
  real(kind=dk), dimension(*), intent(in) :: ui
  real(kind=dk), dimension(6, nc), intent(in) :: d
  real(kind=dk), dimension(nx, nc), intent(inout) :: xtra
  real(kind=dk), dimension(6, nc), intent(inout) :: stressarg
  !.......................................................................local
  logical :: eosmod, softening, elastic
  integer(kind=ik) :: ic, flg, is, ns, conv
  integer(kind=ik), parameter :: ms = 100
  real(kind=dk), parameter, dimension(6) :: &
       delta = (/one, one, one, zero, zero, zero/)
  real(kind=dk), parameter :: tol1=1.d-8, tol2=1.d-6, refeps=.01_dk
  real(kind=dk) :: k, mu, y0, f, a, c0, c1, c2, yld
  real(kind=dk) :: yf, gf, sf, refj2, tolfac, ctp
  real(kind=dk) :: gam, dmgn, epv, enrgy, tmpr, rho
  real(kind=dk), dimension(6) :: de, stress, bstress, dep
  real(kind=dk), dimension(6) :: g, xi, xit, xin, dxit
  character*13 iam
  parameter(iam='futalafu_calc' )
  !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ futalafu_calc

  ! user properties
  y0 = ui(ipy0); a = ui(ipa);
  c0 = ui(ipc0); c1 = ui(ipc1); c2 = ui(ipc2);

  ! reference value of j2
  refj2 = two * ui(ipmu) * refeps

  ! damage properties
  yf = ui(ipyf); gf = ui(ipgf); sf = ui(ipsf);

  ! logicals
  eosmod = ui(ipeosid) /= zero; softening = ui(ipy0) > ui(ipyf);

  gather_scatter: do ic = 1, nc

     ! initialize local variables
     ns = 1
     elastic = xtra(kyld, ic) == zero

     ! if the step starts elastic and ends inelastic, we return to 15 for
     ! subcycling
15   continue

     ! get passed arguments
     de = d(1:6, ic) * dt / float(ns)
     gam = xtra(kgam, ic); epv = xtra(kepv, ic); dmgn = xtra(kdmg, ic);
     bstress = xtra(kbsxx:kbsxz, ic); stress = stressarg(1:6, ic);
     enrgy = xtra(kenrgy, ic); tmpr = xtra(ktmpr, ic);
     rho = xtra(krho, ic) * exp(-tr(de))
     yld = xtra(kyld, ic)

     tolfac = one
     if(dmgn > zero) tolfac = 100._dk

     subcycle: do is = 1, ns

        ! shifted stress at beginning of step
        xin = stress - bstress

        ! elastic moduli
        call elastic_moduli(ui, gam, xin, enrgy, tmpr, rho, k, mu)

        ! elastic predictor
        dxit = three * k * iso(de) + two * mu * dev(de)
        xit = xin + dxit

        ! evaluate yield function at trial state
        call yield_function(flg, xit, gam, f, ctp, g)

        if(f / refj2 <= tolfac * tol1) then
           ! elastic
           yld = zero
           conv = 1
           xi = xit
           dep = zero

        else if(elastic) then
           ! the step started elastic, increase subcycles to sneak up on the
           ! point at which inelasticity occurs
           elastic = .false.
           ns = 10
           go to 15

        else
           ! inelastic step
           elastic = .false.
           call return_stress(xin, xit, gam, yld, conv, xi, dep)
           if(conv == 0) then
              ! return_stress did not converge, if there is softening,
              ! increase the number of sybcycle steps to see if this will
              ! aleviate the problem, if not, quit.
              if(softening) then
                 if(ns < ms) then
                    ns = ns * 10
                    go to 15
                 else
                    ! Increasing subcycles did not aleviate the problem. This
                    ! likely occurred due to excessive curvature in the yield
                    ! function due to softening during the step. Rather than
                    ! quit, issue a warning and *hope* that after a step or
                    ! two the yield function behaves better -> which it should
                    ! once we get to the fully failed state.
                    call logmes("WARNING: Excessive softening has occurred")
                 end if
              else
                 call bombed("Failed to return stress to yield surface")
              end if
           end if
        end if

        ! update back stress
        ! Since backstress is the origin for elasticity, as dmg -> 1,
        ! backstress needs to -> 0.
        bstress = (one - damage(gam)) * (bstress + two / three * a * dev(dep))
        stress = xi + bstress
        enrgy = enrgy + one / rho * ddp(stress, de - dep)
        epv = epv + tr(dep)
     end do subcycle

     ! update passed quantities
     stressarg(1:6, ic) = stress
     xtra(kgam, ic) = gam
     xtra(kepv, ic) = epv
     xtra(kdmg, ic) = damage(gam)
     xtra(kbsxx:kbsxz, ic) = bstress
     xtra(krho, ic) = rho
     xtra(ktmpr, ic) = tmpr
     xtra(kenrgy, ic) = enrgy
     xtra(kr, ic) = mag(dev(stress))
     xtra(kz, ic) = ddp(stress, delta) / sqrt(three)
     xtra(kyld, ic) = yld

     xtra(kf01, ic) = float(conv)

  end do gather_scatter

  return

contains

  !***************************************************************************!

  subroutine elastic_moduli(ui, gam, sig, enrgy, tmpr, rho, k, mu)
    !*************************************************************************!
    !
    ! Compute tangent moduli
    !
    !*************************************************************************!
    implicit none
    !....................................................................passed
    real(kind=dk), intent(in) :: enrgy, rho, gam
    real(kind=dk), intent(inout) :: tmpr
    real(kind=dk), intent(out) :: k, mu
    real(kind=dk), dimension(6), intent(in) :: sig
    real(kind=dk), dimension(*), intent(in) :: ui
    !.....................................................................local
    real(kind=dk) :: dmg_fac, sfratio
    !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ elastic_moduli
    if(eosmod) then
       call eos_moduli(ui, sig, enrgy, tmpr, rho, k, mu)
    else
       k = ui(ipk); mu = ui(ipmu);
    end if

    if(softening) then
       !sfratio = max(0.074d0, ui(ipyf) / ui(ipy0))
       sfratio = max(0.3d0, ui(ipyf) / ui(ipy0))
       dmg_fac = sfratio + (one - sfratio) * (one - damage(gam))
       if(tr(xin) > zero) k = k * dmg_fac
       mu = mu * dmg_fac
    end if
    return
  end subroutine elastic_moduli

  !***************************************************************************!

  subroutine yield_function(surf, sig, gam, f, pmax, g)
    !*************************************************************************!
    !
    ! Evaluate the yield function and gradient
    !
    !*************************************************************************!
    implicit none
    !....................................................................passed
    integer(kind=ik), intent(out) :: surf
    real(kind=dk), intent(in) :: gam
    real(kind=dk), intent(out) :: f, pmax
    real(kind=dk), dimension(6), intent(in) :: sig
    real(kind=dk), dimension(6), intent(out), optional :: g
    !.....................................................................local
    real(kind=dk), parameter :: puny=1.d-16
    real(kind=dk), dimension(6) :: s, gs
    real(kind=dk) :: i1, eqps, rtj2, y, ph, dam
    !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ yield_function

    surf = 1
    ! isotropic and deviatoric part of stress
    i1 = tr(sig); s = dev(sig)

    ! current yield stress
    y = y0
    if(softening) then
       dam = damage(gam)
       y = y - y0 * dam + yf * dam
    end if

    ! maximum (negative) pressure
    pmax = huge(one)
    if(c2 /= zero) pmax = (y + c0 * gam ** (one / c1)) / c2 / three

    ! evaluate the yield function f
    !                1 / c1
    !   f = y + c gam        - c I
    !            0              2 1
    eqps = sqrt(two / three) * gam
    rtj2 = sqrt(half) * mag(s)
    f = max(y + c0 * eqps ** (one / c1) - c2 * i1, zero)
    f = rtj2 - f

    if(.not. present(g)) return

    ! gradient of yield function
    !            df             s
    !      g =  ---- = ------------------ + c2 * I
    !           dsig    sqrt(2) * radius
    if(mag(s) < puny) then
       gs = zero
    else
       gs = s / sqrt(two) / mag(s)
    end if
    g = gs + c2 * delta

    ! if pressure dependent yield surface, check vertex
    if(c2 /= zero) then
       ph = tr(sig - pmax * delta) / mag(sig - pmax * delta)
       if(ph > tr(g) / mag(g)) then
          surf = 2
       end if
    end if

    return

  end subroutine yield_function

  !***************************************************************************!

  subroutine return_stress(sigb, sigt, gam, yld, conv, sig, dep)
    !*************************************************************************!
    !
    ! Return the stress to the yield surface
    ! numerically
    !
    !*************************************************************************!
    implicit none
    !....................................................................passed
    real(kind=dk), dimension(6), intent(in) :: sigb, sigt
    real(kind=dk), intent(inout) :: gam
    integer(kind=ik), intent(out) :: conv
    real(kind=dk), intent(out) :: yld
    real(kind=dk), dimension(6), intent(out) :: sig, dep
    !.....................................................................local
    integer(kind=ik) :: i, side
    integer(kind=ik), parameter :: maxit1=30, maxit2=50
    real(kind=dk) :: f, dlam, df, gamn, sfac, tfac
    real(kind=dk) :: xs, xt, xr, fs, ft, fr, gt, gr, ctp
    real(kind=dk), dimension(6) :: g, n, p, m
    real(kind=dk), parameter :: small=1.d-8
    !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ return_stress

    ! step is at least partially inelastic

    ! initialize local variables
    gamn = gam
    dlam = zero

    ! evaluate yield function
    call yield_function(flg, sigt, gam, fs, ctp, g)

    if(flg == 2) then
       ! vertex
       call return_stress_to_vertex(sigb, ctp, gam, yld, sig, dep)
       conv = 2
       return
    end if

    n = g / mag(g)
    m = n
    p = three * k * iso(m) + two * mu * dev(m)
    yld = one

    if(softening) then

       ! Find initial best guess for dlam by a regula falsi method. We set the
       ! intial brackets on dlam to be
       !
       !                         dlam E [0, mag(de)]
       !
       ! and shrink the bracket size through n steps. When dlam is found, it
       ! is then passed on to the Newton solver that finishes the job. In the
       ! below algorithm, the variable <x> is dlam, <g> is gam, and <f> is the
       ! value of the yield function. The postfixes "s", "t", and "r" stand
       ! for:
       !
       !        s -> over yield, t -> under yield, r -> interpolated
       side = 0; sfac = one; tfac = one
       xs = zero
       xt = mag(de)
       gt = gamn + mag(dev(de))
       call yield_function(flg, sigt - xt * p, gt, ft, ctp, g)

       do i = 1, 10
          xr = (fs / sfac * xt - ft / tfac * xs) / (fs / sfac - ft / tfac)
          if(abs(xt - xs) < small * abs(xt + xs)) then
             exit
          end if

          gr = gamn + mag(dev(xr * m))
          call yield_function(flg, sigt - xr * p, gr, fr, ctp, g)

          sfac = one; tfac = one
          if(fr * ft > zero) then
             xt = xr; ft = fr;
             if(side == -1) sfac = one / two
             side = -1
          else if(fs * fr > zero) then
             xs = xr; fs = fr;
             if(side == 1) tfac = one / two
             side = 1
          else
             exit
          end if

       end do
       dlam = max(xs, xr); gam = gamn + mag(dev(dlam * m));

    end if

    ! Newton iterations to find plastic strain increment
    conv = 0
    newton: do i = 1, maxit2

       ! updated stress
       sig = sigt - dlam * p

       ! evaluate yield function at updated stress
       call yield_function(flg, sig, gam, f, ctp, g)

       ! check convergence
       if(i <= maxit1) then
          if(abs(f) / refj2 < tolfac * tol1) then
             conv = 1
             exit newton
          end if
       else
          if(abs(f) / refj2 < tolfac * tol2) then
             conv = 2
             exit newton
          end if
       end if

       ! check if maximum iterations exceeded
       if(i > maxit2) exit newton

       ! perform the Newton step
       df = yield_function_deriv(sig, m, p, gam, dlam, f)
       dlam = dlam - f / df
       gam = gamn + mag(dev(dlam * m))

    end do newton
    dep = dlam * m
    gam = gamn + mag(dev(dep))

  end subroutine return_stress

  !***************************************************************************!

  subroutine return_stress_to_vertex(sigb, ctp, gam, yld, sig, dep)
    !*************************************************************************!
    !
    ! Return the stress to the yield surface
    ! numerically
    !
    !*************************************************************************!
    implicit none
    !....................................................................passed
    real(kind=dk), intent(in) :: ctp
    real(kind=dk), dimension(6), intent(in) :: sigb
    real(kind=dk), intent(inout) :: gam
    real(kind=dk), intent(out) :: yld
    real(kind=dk), dimension(6), intent(out) :: sig, dep
    !.....................................................................local
    integer(kind=ik) :: i, side
    real(kind=dk) :: gamn
    real(kind=dk) :: xs, xt, xr, fs, ft, fr, gr
    real(kind=dk), dimension(6) :: dsig, dsigr, sigr
    real(kind=dk), parameter :: small=1.d-8
    !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ return_stress

    ! yld = 2 -> on vertex
    yld = two
    gamn = gam

    sig = ctp * delta
    dsig = sigb - sig
    dep = de - one / three / k * iso(dsig) - one / two / mu * dev(dsig)
    gam = gamn + mag(dev(dep))
    call yield_function(flg, sig, gam, fs, xs)

    if(softening .and. abs(fs) / refj2 > tolfac * tol1) then
       ! Find the softened stress using a regula falsi method.
       ! The stress has already been returned to the vertex before softening,
       ! but as the surface softens due to the additional plastic strain, the
       ! updated stress will be beyond yield. Here, we bracket the stress
       ! between the value at the fully failed state and the current and
       ! decrease the range of the bracketed stress until within a reasonable
       ! tolerance.
       side = 0

       ! this first call is with a large value of the magnitude of the
       ! distortional plastic strain to get the value of ctp for the fully
       ! failed surface.
       call yield_function(flg, sig, ten * gf, ft, xt)

       conv = 0
       do i = 1, 20
          xr = (fs * xt - ft * xs) / (fs - ft)
          if(abs(xt - xs) < small * abs(xt + xs)) then
             conv = 1
             exit
          end if

          sigr = xr * delta
          dsigr = sigr - sigb
          gr = gamn + mag(dev(&
               de - one / three / k * iso(dsigr) - one / two / mu * dev(dsigr)))
          call yield_function(flg, sigr, gr, fr, xr)

          if(fr * ft > zero) then
             xt = xr; ft = fr;
             if(side == -1) fs = fs / two
             side = -1
          else if(fs * fr > zero) then
             xs = xr; fs = fr;
             if(side == 1) ft = ft / two
             side = 1
          else
             conv = -1
             exit
          end if
          conv = 2
       end do

       if(conv == -1) call bombed("Unable to bracket softened ctp")

       ! updated stress found
       sig = xs * delta
       dsig = sig - sigb
       dep = de - one / three / k * iso(dsig) - one / two / mu * dev(dsig)
       gam = mag(dev(dep))
    end if

    return
  end subroutine return_stress_to_vertex

  !***************************************************************************!

  function yield_function_deriv(sig, m, p, gam, dlam, f)
    !*************************************************************************!
    !
    ! Calculate the derivative of the yield function with respect to gamma
    ! numerically
    !
    !*************************************************************************!
    implicit none
    !....................................................................passed
    real(kind=dk) :: yield_function_deriv
    real(kind=dk), intent(in) :: gam, dlam, f
    real(kind=dk), dimension(6), intent(in) :: sig, m, p
    !.....................................................................local
    integer(kind=ik) :: flg
    real(kind=dk) :: eps, fp, gamp, ctp
    real(kind=dk), dimension(6) :: sigp
    !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ yield_function_deriv
    ! perturb dgam, sig, and gam by eps
    eps = sqrt(epsilon(dlam))
    sigp = sig - (dlam + eps) * p
    gamp = gam + mag(dev((dlam + eps) * m))
    call yield_function(flg, sigp, gam, fp, ctp)
    yield_function_deriv = (fp - f) / eps
    return
  end function yield_function_deriv

  !***************************************************************************!

  function damage(gam)
    !*************************************************************************!
    !
    ! Accumulate damage
    !
    !*************************************************************************!
    implicit none
    !....................................................................passed
    real(kind=dk) :: damage
    real(kind=dk), intent(in) :: gam
    !.....................................................................local
    real(kind=dk) :: coher, ddmg
    !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ damage
    ! calculate the coherence
    coher = (one + exp(-two * sf)) / (one + exp(-two * sf * (one - gam / gf)))

    ! The evolution of damage evolution has a couple of rules:
    !    1) No healing -> dmgp >= dmgn
    damage = max(one - coher, dmgn)

    !    2) Maximum change of 15% allowed in any one step
    !       12.5% was chosen out of a hat!
    ddmg = min(damage - dmgn, .15_dk)
    damage = dmgn + ddmg
    return
  end function damage

end subroutine futalafu_calc
