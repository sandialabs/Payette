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


!*******************************************************************************!

module futalafu_constants
  integer, parameter :: dk=selected_real_kind(14)

  ! parameter pointers
  integer, parameter :: ipk = 1
  integer, parameter :: ipmu = 2
  integer, parameter :: ipnu = 3
  integer, parameter :: ipy0 = 4
  integer, parameter :: ipa = 5
  integer, parameter :: ipc0 = 6
  integer, parameter :: ipc1 = 7
  integer, parameter :: ipc2 = 8
  integer, parameter :: ipyf = 9
  integer, parameter :: ipgf = 10
  integer, parameter :: ipsf = 11
  integer, parameter :: ipf01 = 12
  integer, parameter :: ipeosid = 13
  integer, parameter :: ipr0 = 14
  integer, parameter :: ipt0 = 15
  integer, parameter :: ipcs = 16
  integer, parameter :: ipcv = 19

  ! state variable pointers
  integer, parameter :: nsv=15
  integer, parameter :: kgam = 1
  integer, parameter :: kepv = 2
  integer, parameter :: kdam = 3
  integer, parameter :: kbs = 3
  integer, parameter :: kbsxx = kbs+1
  integer, parameter :: kbsyy = kbs+2
  integer, parameter :: kbszz = kbs+3
  integer, parameter :: kbsxy = kbs+4
  integer, parameter :: kbsyz = kbs+5
  integer, parameter :: kbsxz = kbs+6
  integer, parameter :: ktherm = kbsxz
  integer, parameter :: kenrgy = ktherm+1
  integer, parameter :: ktmpr = ktherm+2
  integer, parameter :: krho = ktherm+3
  integer, parameter :: kmech = ktherm+3
  integer, parameter :: kr = kmech+1
  integer, parameter :: kz = kmech+2
  integer, parameter :: kyld = nsv

  ! numbers
  real(kind=dk), parameter :: half=.5_dk
  real(kind=dk), parameter :: zero=0._dk, one=1._dk, two=2._dk
  real(kind=dk), parameter :: three=3._dk, six=6._dk, ten=10._dk

end module futalafu_constants


subroutine futalafu_chk(ui)
  !*****************************************************************************!
  !
  ! Checks validity of user inputs
  ! Sets defaults for unspecified user input.
  ! Adjusts user input to be self-consistent.
  !
  !*****************************************************************************!
  use futalafu_constants
  implicit none
  !....................................................................parameters
  !........................................................................passed
  real(kind=dk), dimension (*) :: ui
  !.........................................................................local
  logical :: eosmod
  character*12 iam
  parameter(iam='futalafu_chk' )
  ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ futalafu_chk
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
  if(ui(ipgf) <= zero) ui(ipgf) = 1.d60

  ! poisson's ratio
  ui(ipnu) = (three * ui(ipk) - two * ui(ipmu)) / (six * ui(ipk) + two * ui(ipmu))
  if(ui(ipnu) < zero) call logmes("#---- WARNING: negative Poisson's ratio")

  ! check the equation of state and energetic properties
  if(ui(ipt0) <= zero) ui(ipt0) = 298._dk
  if(eosmod) call check_eos(ui)

  return
end subroutine futalafu_chk


!*******************************************************************************!


subroutine futalafu_rxv(ui, nx, namea, keya, rinit, iadvct)
  !*****************************************************************************!
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
  !*****************************************************************************!
  use futalafu_constants
  implicit none

  !....................................................................parameters
  integer, parameter :: mmcn=50, mmck=10, mnunit=7
  integer, parameter :: mmcna=nsv*mmcn, mmcka=nsv*mmck
  !........................................................................passed
  integer :: nx
  integer, dimension(*) :: iadvct
  real(kind=dk), dimension(*) :: ui, rinit
  character*1 namea(*), keya(*)

  !.........................................................................local
  character*(mmcn) name(nsv)
  character*(mmck) key(nsv)
  ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ futalafu_rxv

  call logmes('#---- requesting futalafu extra variables')

  rinit(1:nsv) = zero

  nx = 0
  ! distortional plastic strain
  nx = nx+1
  name(nx) = 'distortional plastic strain'
  key(nx) = 'GAM'
  iadvct(nx) = 0        ! input and output

  ! plastic volume strain
  nx = nx+1
  name(nx) = 'plastic volume strain'
  key(nx) = 'EPV'
  iadvct(nx) = 0        ! input and output

  ! damage
  nx = nx+1
  name(nx) = 'damage'
  key(nx) = 'DAM'
  iadvct(nx) = 0        ! input and output

  ! -- back stress
  nx=nx+1
  name(nx)='11 component of back stress'
  key(nx)='BSIG11'
  iadvct(nx) = 1

  nx=nx+1
  name(nx)='22 component of back stress'
  key(nx)='BSIG22'
  iadvct(nx) = 1

  nx=nx+1
  name(nx)='33 component of back stress'
  key(nx)='BSIG33'
  iadvct(nx) = 1

  nx=nx+1
  name(nx)='12 component of back stress'
  key(nx)='BSIG12'
  iadvct(nx) = 1

  nx=nx+1
  name(nx)='23 component of back stress'
  key(nx)='BSIG23'
  iadvct(nx) = 1

  nx=nx+1
  name(nx)='13 component of back stress'
  key(nx)='BSIG13'
  iadvct(nx) = 1

  ! specific energy
  nx=nx+1
  name(nx)='Specific energy'
  key(nx)='ENRGY'
  iadvct(nx) = 1
  rinit(nx)=ui(ipcv) * ui(ipt0)

  ! temperature
  nx=nx+1
  name(nx)='Temperature'
  key(nx)='TMPR'
  iadvct(nx) = 1
  rinit(nx)=ui(ipt0)

  ! density
  nx=nx+1
  name(nx)='Density'
  key(nx)='RHO'
  iadvct(nx) = 1
  rinit(nx)=ui(ipr0)

  ! magnitude of deviatoric stress
  nx=nx+1
  name(nx)="Magnitude of deviatoric stress"
  key(nx)="R"
  iadvct(nx) = 1

  ! magnitude of hydrostatic stress
  nx=nx+1
  name(nx)="Magnitude of hydrostatic stress"
  key(nx)="Z"
  iadvct(nx) = 1

  ! yield surface flag
  nx=nx+1
  name(nx)="Yield flag"
  key(nx)="YLD"
  iadvct(nx) = 1

  call tokens(nx, name, namea)
  call tokens(nx, key, keya)
  return
end subroutine futalafu_rxv


!*******************************************************************************!


subroutine futalafu_calc(nc, nxtra, dt, ui, stress, d, xtra)
  !*****************************************************************************!
  !
  ! Combined kinematic/isotropic hardening pressure dependent plasticity
  !
  !*****************************************************************************!
  !
  ! input arguments
  ! ===============
  ! nc         int                   Number of blocks to be processed
  ! nxtra      int                   Number of internal state vars
  ! dt         dp                    Current time increment
  ! ui         dp,ar(nprop)          User inputs
  ! d          dp,ar(6)              Strain increment
  !
  ! input output arguments
  ! ======================
  ! stress   dp,ar(6)                stress
  ! xtra     dp,ar(nxtra)            state variables
  !
  ! output arguments
  ! ================
  ! usm      dp                      uniaxial strain modulus
  !
  !*****************************************************************************!
  !
  ! stresss and strains, plastic strain tensors
  ! 11, 22, 33, 12, 23, 13
  !
  !*****************************************************************************!
  use tensor_toolkit
  use futalafu_constants
  implicit none
  !........................................................................passed
  integer :: nc, nxtra
  real(kind=dk) :: dt
  real(kind=dk), dimension(*) :: ui
  real(kind=dk), dimension(nxtra, nc) :: xtra
  real(kind=dk), dimension(6, nc) :: stress, d
  !........................................................................local
  integer :: ic
  logical :: eosmod, damage
  real(kind=dk), parameter, dimension(6) :: &
       delta = (/one, one, one, zero, zero, zero/)
  real(kind=dk), parameter :: ftol=1.d-6
  real(kind=dk) :: k, mu, y0, f, a, c0, c1, c2, twomu, threek, yld
  real(kind=dk) :: yf, gf, sf, y
  real(kind=dk) :: rtj2, cti1
  real(kind=dk) :: dgam, gam, epv, dam, enrgy, tmpr, rho
  real(kind=dk), dimension(6) :: de, sig, bstress, dep
  real(kind=dk), dimension(6) :: g, xid, xi, xin, dxi
  character*13 iam
  parameter(iam='futalafu_calc' )
  ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ futalafu_calc

  ! user properties
  k = ui(ipk); mu = ui(ipmu); y0 = ui(ipy0); a = ui(ipa);
  c0 = ui(ipc0); c1 = ui(ipc1); c2 = ui(ipc2);

  ! damage properties
  yf = ui(ipyf); gf = ui(ipgf); sf = ui(ipsf);

  ! logicals
  eosmod = ui(ipeosid) /= zero; damage = y0 /= yf;

  gather_scatter: do ic = 1, nc

     ! get passed arguments
     de = d(1:6, ic) * dt;
     gam = xtra(kgam, ic); epv = xtra(kepv, ic); dam = xtra(kdam, ic);
     bstress = xtra(kbsxx:kbsxz, ic); sig = stress(1:6, ic);
     enrgy = xtra(kenrgy, ic); tmpr = xtra(ktmpr, ic);
     rho = xtra(krho, ic) * exp(-(de(1) + de(2) + de(3)))

     ! shifted stress
     xin = sig - bstress

     ! initialize local variables
     yld = zero

     ! current yield stress
     y = y0
     if(damage) y = y - y0 * dam + yf * dam

     ! peak value of I1
     cti1 = 1.d99
     if(c2 /= zero) cti1 = (y + c0 * gam ** (one / c1)) / c2

     ! elastic moduli
     if(eosmod) call get_eos_moduli(ui, sig, enrgy, tmpr, rho, k, mu)
     threek = three * k
     if(tr(de) > zero) threek = threek * (one - .95_dk * dam)
     twomu = two * mu

     ! elastic predictor
     dxi = threek * iso(de) + twomu * dev(de)
     xi = xin + dxi
     xid = dev(xi)

     ! evaluate yield function
     rtj2 = sqrt(half) * mag(xid)
     call evaluate_yield_function(0, tr(xi), xid, gam, f, g)

     if(rtj2 - f > ftol) then

        call return_stress(xi, gam, yld)

        if(yld < zero) call bombed("Newtwon iterations failed")

        ! determine plastic strain
        dxi = xi - xin
        dep = de - one / threek * iso(dxi) + one / twomu * dev(dxi)
        dgam = mag(dev(dep))
        gam = gam + dgam
        epv = epv + tr(dep)

        ! update back stress
        ! Since backstress is now origin for elasticity, as dam -> 1,
        ! backstress needs to -> 0.
        bstress = (one - dam) * (bstress + two / three * a * dev(dep))

        if(damage) then
           ! update damage
           call update_damage(gam, dam)
           !      call return_stress(xi, gam, yld)
        end if

     end if

     ! update passed quantities
     stress(1:6, ic) = xi + bstress
     xtra(kgam, ic) = gam
     xtra(kepv, ic) = epv
     xtra(kdam, ic) = dam
     xtra(kbsxx:kbsxz, ic) = bstress
     xtra(krho, ic) = rho
     xtra(ktmpr, ic) = tmpr
     xtra(kenrgy, ic) = enrgy + one / rho * ddp(sig, de)
     xtra(kr, ic) = mag(dev(sig))
     xtra(kz, ic) = ddp(sig, delta) / sqrt(three)
     xtra(kyld, ic) = yld

  end do gather_scatter

  return

  contains

    !***************************************************************************!

    subroutine evaluate_yield_function(flg, i1, s, gam, f, g)
      !*************************************************************************!
      !
      ! Evaluate the yield function and gradient
      !
      !*************************************************************************!
      implicit none
      !....................................................................passed
      integer :: flg
      real(kind=dk) :: i1, gam, f
      real(kind=dk), dimension(6) :: s, g
      !.....................................................................local
      real(kind=dk), parameter :: puny=1.d-16
      real(kind=dk), dimension(6) :: gs
      real(kind=dk) :: eqps
      !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ evaluate_yield_function

      ! evaluate the yield function f
      !                1 / c1
      !   f = y + c gam        - c I
      !            0              2 1
      eqps = sqrt(two / three) * gam
      f = max(y + c0 * eqps ** (one / c1) - c2 * i1, zero)

      ! check vertex
      if(i1 > cti1) f = -two

      ! initialize gradient and exit if not requested
      g = zero
      if(abs(flg) < 1) return

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

      return

    end subroutine evaluate_yield_function

    !***************************************************************************!

    subroutine return_stress(sig, gam, yld)
      !*************************************************************************!
      !
      ! return the stress to the limit surface
      !
      !*************************************************************************!
      implicit none
      !....................................................................passed
      real(kind=dk) :: gam, yld
      real(kind=dk), dimension(6) :: sig
      !.....................................................................local
      integer :: i
      real(kind=dk) :: diff, beta, f, pmax, ph
      real(kind=dk), dimension(6) :: p, m, n, s
      real(kind=dk), parameter :: tiny=1.d-14, small=1.d-10
      !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ return_stress

      ! step is at least partially inelastic
      yld = -one

      ! evaluate yield function
      s = dev(sig)
      rtj2 = sqrt(half) * mag(s)
      call evaluate_yield_function(1, tr(sig), s, gam, f, g)
      diff = rtj2 - f

      newton: do i = 1, 25
         ! Perform Newton iterations to find magnitude of projection from
         ! the trial stress state to the yield surface.

         ! yield function normal
         !            df/dsig
         !     n = -------------,  p = C : m
         !          ||df/dsig||

         n = g / mag(g)

         ! flow direction, tentatively assume normality
         m = n

         ! check vertex
         pmax = cti1 / three
         ph = tr(sig - pmax * delta) / mag(sig - pmax * delta)
         if(ph >= tr(m)) then
            yld = two
            sig = pmax * delta
            exit newton
         end if

         ! return direction
         p = threek * iso(m) + twomu * dev(m)

         ! only the direction of P matters, normalize it so that it is on
         ! the same order of magnitude as stress
         p = sqrt(mag(sig) / mag(p)) * p

         ! apply the Newton-Raphson step
         beta = -diff / ddp(n, p)

         ! improved estimates for the stress
         sig = sig + beta * p
         s = dev(sig)

         if(abs(beta) < tiny .or. (abs(beta) < small .and. diff < ftol)) then
            yld = abs(yld)
            exit newton
         end if

         ! update the trial stress and yield function
         rtj2 = sqrt(half) * mag(s)
         call evaluate_yield_function(1, tr(sig), s, gam, f, g)
         diff = rtj2 - f

      end do newton

    end subroutine return_stress

    !***************************************************************************!

    subroutine update_damage(gam, dam)
      !*************************************************************************!
      !
      ! Accumulate damage
      !
      !*************************************************************************!
      implicit none
      !....................................................................passed
      real(kind=dk) :: gam, dam
      !.....................................................................local
      !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ update_damage
      dam = (one + exp(-two * sf)) / (one + exp(-two * sf * (one - gam / gf)))
      dam = one - dam
      return

    end subroutine update_damage

end subroutine futalafu_calc

         ! ! consistency parameter
         ! !                  n : dsig
         ! !         dgam = -----------,  H = dfda : ha + dfdy * hy
         ! !                 n : p - H

         ! ! hardening modulus
         ! ha = two / three * a * dev(m)
         ! dfda = -s / sqrt(two) / mag(s)
         ! hy = zero
         ! if(c0 /= zero) hy = c1 * c0 * ((f - y) / c0) ** ((c1 - 1) / c1)
         ! dfdy = -one / sqrt(three)
         ! H = sqrt(two) * (ddp(dfda, ha) + dfdy * hy)

