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


! ***************************************************************************** !


subroutine plast_chk(ui)
  ! *************************************************************************** !
  !     REQUIRED MIG DATA CHECK ROUTINE
  !     Checks validity of user inputs for DMM model.
  !     Sets defaults for unspecified user input.
  !     Adjusts user input to be self-consistent.
  ! *************************************************************************** !

  implicit none

  !........................................................................passed
  double precision, dimension (*) :: ui
  !.........................................................................local
  double precision :: k, mu, y, a, c, m, nu
  character*9 iam
  parameter(iam='plast_chk' )
  ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ plast_chk

  k = ui(1)
  mu = ui(2)
  y = ui(3)
  a = ui(4)
  c = ui(5)
  m = ui(6)
  if(k .le. 0.d0) &
       call faterr(iam, "Bulk modulus K must be positive")
  if(mu .le. 0.d0) &
       call faterr(iam, "Shear modulus MU must be positive")
  if(y .le. 0.d0) &
       call faterr(iam, "Yield strength Y must be positive")
  if(a .lt. 0.d0) &
       call faterr(iam, "Kinematic hardening modulus A must be non-negative")
  if(c .lt. 0.d0) &
       call faterr(iam, "Isotropic hardening modulus C must be non-negative")
  if(m .lt. 0.d0) then
       call faterr(iam, "Isotropic hardening power M must be non-negative")
  else if(m .eq. 0.d0) then
     if(c .ne. 0.d0) &
          call logmes("Isotropic hardening modulus C being set 0 because M = 0")
     ui(5) = 0.d0
  end if

  ! poisson's ratio
  nu = (3.d0 * k - 2.d0 * mu) / (6.d0 * k + 2.d0 * mu)
  if(nu .lt. 0.d0) &
       call logmes("WARNING: negative Poisson's ratio")

  return
end subroutine plast_chk


! ***************************************************************************** !


subroutine plast_rxv(nx, namea, keya, rinit, iadvct)
  ! *************************************************************************** !
  !     REQUESTED EXTRA VARIABLES FOR KAYENTA
  !
  !     This subroutine creates lists of the internal state variables
  !     needed for DMM. This routine merely sends a
  !     LIST of internal state variable requirements back to the host
  !     code.   IT IS THE RESPONSIBILITY OF THE HOST CODE to loop over
  !     the items in each list to actually establish necessary storage
  !     and (if desired) set up plotting, restart, and advection
  !     control for each internal state variable.
  !
  !     called by: host code after all input data have been checked
  ! *************************************************************************** !

  implicit none

  !....................................................................parameters
  integer, parameter :: nsv=7
  integer, parameter :: mmcn=50, mmck=10, mnunit=7
  integer, parameter :: mmcna=nsv*mmcn, mmcka=nsv*mmck

  !........................................................................passed
  integer :: nx
  integer, dimension(*) :: iadvct
  double precision, dimension(*) :: rinit
  character*1 namea(*), keya(*)

  !.........................................................................local
  character*(mmcn) name(nsv)
  character*(mmck) key(nsv)
  ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ plast_rxv

  call logmes('############# requesting plastic extra variables')

  rinit(1:nsv) = 0.d0

  nx = 0
  ! equivalent plastic strain
  nx = nx+1
  name(nx) = 'distortional plastic strain'
  key(nx) = 'GAM'
  iadvct(nx) = 0        ! input and output

  ! -- back stress
  nx=nx+1
  name(nx)='11 component of back stress'
  key(nx)='BSIG11'
  iadvct(nx)=1

  nx=nx+1
  name(nx)='22 component of back stress'
  key(nx)='BSIG22'
  iadvct(nx)=1

  nx=nx+1
  name(nx)='33 component of back stress'
  key(nx)='BSIG33'
  iadvct(nx)=1

  nx=nx+1
  name(nx)='12 component of back stress'
  key(nx)='BSIG12'
  iadvct(nx)=1

  nx=nx+1
  name(nx)='23 component of back stress'
  key(nx)='BSIG23'
  iadvct(nx)=1

  nx=nx+1
  name(nx)='13 component of back stress'
  key(nx)='BSIG13'
  iadvct(nx)=1

  call tokens(nx, name, namea)
  call tokens(nx, key, keya)
  return
end subroutine plast_rxv


!****************************************************************************** !


subroutine plast_calc(nc, nsv, dt, ui, sigarg, darg, svarg)
  !**************************************************************************** !
  !
  !     Description:
  !       Combined kinematic/isotropic hardening plasticity
  !
  !**************************************************************************** !
  !
  !     input arguments
  !     ===============
  !      NBLK       int                   Number of blocks to be processed
  !      NINSV      int                   Number of internal state vars
  !      DT         dp                    Current time increment
  !      UI         dp,ar(nprop)          User inputs
  !      D          dp,ar(6)              Strain increment
  !
  !     input output arguments
  !     ======================
  !      STRESS   dp,ar(6)                stress
  !      SV       dp,ar(nsv)            state variables
  !
  !     output arguments
  !     ================
  !      USM      dp                      uniaxial strain modulus
  !
  !**************************************************************************** !
  !
  !      stresss and strains, plastic strain tensors
  !          11, 22, 33, 12, 23, 13
  !
  !**************************************************************************** !

  implicit none

  !....................................................................parameters
  double precision, parameter, dimension(6) :: delta = (/1.,1.,1.,0.,0.,0./)
  double precision, parameter, dimension(6) :: w = (/1.,1.,1.,2.,2.,2./)
  !........................................................................passed
  integer :: nc, nsv
  double precision :: dt
  double precision, dimension(*) :: ui
  double precision, dimension(nsv, nc) :: svarg
  double precision, dimension(6, nc) :: sigarg, darg
  !........................................................................local
  integer :: ic
  double precision :: k, mu, y0, y, a, c, m, twomu, threek, rt2j2
  double precision :: gam, facyld, dfdy, hy, h, radius
  double precision :: dlam, num, dnom
  double precision, dimension(6) :: de, sig, dsig, bstress, xid, xi, n, p
  double precision, dimension(6) :: dfda, ha
  character*10 iam
  parameter(iam='plast_calc' )
  ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ plast_calc

  ! user properties
  k = ui(1)
  mu = ui(2)
  y0 = ui(3)
  a = ui(4)
  c = ui(5)
  m = ui(6)

  ! constants
  twomu = 2.d0 * mu
  threek = 3.d0 * k

  gather_scatter: do ic = 1, nc

     ! get passed arguments
     de = darg(1:6, ic) * dt
     gam = svarg(1, ic)
     bstress = svarg(2:7, ic)

     ! elastic predictor
     dsig = threek * iso(de) + twomu * dev(de)
     sig = sigarg(1:6, ic) + dsig

     ! elastic predictor relative to back stress - shifted stress
     xi = sig - bstress

     ! deviator of shifted stress and its magnitude
     xid = dev(xi)
     rt2j2 = mag(xid)

     ! yield stress
     y = y0
     if(c .ne. 0.d0) y = y + c * gam ** (1 / m)
     radius = sqrt(2.d0 / 3.d0) * y

     ! check yield
     facyld = 0.d0
     if(rt2j2 - radius .ge. 0.d0) facyld = 1.d0
     rt2j2 = rt2j2 + (1.d0 - facyld) ! avoid any divide by zero

     ! yield surface normal and return direction
     !                   df/dsig
     !            n = -------------,  p = C : n
     !                 ||df/dsig||
     !
     !            df           xid         || df ||      1
     !           ---- = ----------------,  ||----|| = -------
     !           dsig    root2 * radius    ||dsig||    root2
     n = xid / rt2j2 ! radius
     p = threek * iso(n) + twomu * dev(n)

     ! consistency parameter
     !                  n : dsig
     !         dlam = -----------,  H = dfda : ha + dfdy * hy
     !                 n : p - H

     ! numerator
     num = ddp(n, dsig)

     ! denominator
     ha = 2.d0 / 3.d0 * a * dev(n)
     dfda = -xid / sqrt(2.d0) / rt2j2 ! radius
     hy = 0.d0
     if(c .ne. 0.d0) hy = m * c * ((y - y0) / c) ** ((m - 1) / m)
     dfdy = -1.d0 / sqrt(3.d0)
     H = sqrt(2.d0) * (ddp(dfda, ha) + dfdy * hy)
     dnom = ddp(n, p) - H + (1.d0 - facyld) ! avoid any divide by zero

     dlam = facyld * num / dnom
     if(dlam .lt. 0.d0) call bombed(iam//": negative dlam")

     ! equivalet plastic strain
     gam = gam + dlam * mag(dev(n))

     ! update back stress
     bstress = bstress + 2.d0 / 3.d0 * a * dlam * dev(n)

     ! update stress
     sig = sig - dlam * p

     ! store data
     sigarg(1:6, ic) = sig
     svarg(1, ic) = gam
     svarg(2:7, ic) = bstress

  end do gather_scatter

  return

  contains

    function ddp(a, b)
      implicit none
      double precision ddp
      double precision, dimension(6) :: a, b
      ddp = sum(w * a * b)
      return
    end function ddp

    function mag(a)
      implicit none
      double precision mag
      double precision, dimension(6) :: a
      mag = sqrt(ddp(a, a))
      return
    end function mag

    function dev(a)
      implicit none
      double precision, dimension(6) :: dev, a
      dev = a - iso(a)
      return
    end function dev

    function iso(a)
      implicit none
      double precision, dimension(6) :: iso, a
      iso = ddp(a, delta) / 3. * delta
      return
    end function iso

end subroutine plast_calc
