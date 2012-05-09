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

subroutine plast_chk(ui)
  !***********************************************************************
  !     REQUIRED MIG DATA CHECK ROUTINE
  !     Checks validity of user inputs for DMM model.
  !     Sets defaults for unspecified user input.
  !     Adjusts user input to be self-consistent.
  !
  !***********************************************************************

  implicit none

  !...................................................................... passed
  double precision, dimension (*) :: ui
  !...................................................................... local
  double precision :: k, mu, y, a, c, m, nu
  character*9 iam
  parameter(iam='plast_chk' )

  ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ plast_chk

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

subroutine plast_rxv(nx, namea, keya, rinit, iadvct)
!**********************************************************************
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
!
!***********************************************************************

  implicit none

  !.................................................................. parameters
  integer, parameter :: nsv=7
  integer, parameter :: mmcn=50, mmck=10, mnunit=7
  integer, parameter :: mmcna=nsv*mmcn, mmcka=nsv*mmck

  !...................................................................... passed
  integer :: nx
  integer, dimension(*) :: iadvct
  double precision, dimension(*) :: rinit
  character*1 namea(*), keya(*)

  !...................................................................... local
  character*(mmcn) name(nsv)
  character*(mmck) key(nsv)

  !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ plast_rxv
  call logmes('############# requesting plastic extra variables')

  rinit(1:nsv) = 0.d0

  nx = 0
  ! equivalent plastic strain
  nx = nx+1
  name(nx) = 'equivalent plastic strain'
  key(nx) = 'EQPS'
  iadvct(nx) = 0        ! input and output

  ! -- back stress
  nx=nx+1
  name(nx)='11 component of back stress'
  key(nx)='QSSIG11'
  iadvct(nx)=1

  nx=nx+1
  name(nx)='22 component of back stress'
  key(nx)='QSSIG22'
  iadvct(nx)=1

  nx=nx+1
  name(nx)='33 component of back stress'
  key(nx)='QSSIG33'
  iadvct(nx)=1

  nx=nx+1
  name(nx)='12 component of back stress'
  key(nx)='QSSIG12'
  iadvct(nx)=1

  nx=nx+1
  name(nx)='23 component of back stress'
  key(nx)='QSSIG23'
  iadvct(nx)=1

  nx=nx+1
  name(nx)='13 component of back stress'
  key(nx)='QSSIG13'
  iadvct(nx)=1

  call tokens(nx, name, namea)
  call tokens(nx, key, keya)
  return
end subroutine plast_rxv






subroutine plast_calc(nc, nsv, dt, ui, sigarg, darg, svarg)
  !***********************************************************************
  !
  !     Description:
  !       Combined kinematic/isotropic hardening plasticity
  !
  !***********************************************************************
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
  !***********************************************************************
  !
  !      stresss and strains, plastic strain tensors
  !          11, 22, 33, 12, 23, 13
  !
  !***********************************************************************

  implicit none

  !.................................................................. parameters
  double precision, parameter :: root23=sqrt(2.d0/3.d0), two_third=2.d0/3.d0
  double precision, parameter, dimension(6) :: delta = (/1.,1.,1.,0.,0.,0./)
  double precision, parameter, dimension(6) :: w = (/1.,1.,1.,2.,2.,2./)

  !...................................................................... passed
  integer :: nc, nsv
  double precision :: dt
  double precision, dimension(*) :: ui
  double precision, dimension(nsv, nc) :: svarg
  double precision, dimension(6, nc) :: sigarg, darg
  !...................................................................... local
  integer :: ic
  double precision :: k, mu, y0, y, a, c, m, twomu, alam, smean, trde, dsmag
  double precision :: eqps, facyld, nddp, h_kin, h_iso, dnom, diff, radius, dlam
  double precision :: fac
  double precision, dimension(6) :: de, sig, bstress, ds, s

  ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ plast_chk

  ! user properties
  k = ui(1)
  mu = ui(2)
  y0 = ui(3)
  a = ui(4) * two_third
  c = ui(5)
  m = ui(6)

  ! constants
  twomu = 2. * mu
  alam = k - twomu / 3.

  gather_scatter: do ic = 1, nc

     ! get passed arguments
     de = darg(1:6, ic) * dt
     eqps = svarg(1, ic)
     bstress = svarg(2:7, ic)

     ! elastic predictor
     trde = sum(de * delta)
     sig = sigarg(1:6, ic) + alam * trde * delta + twomu * de

     ! elastic predictor relative to back stress - shifted stress
     s = sig - bstress

     ! deviator of shifted stress and its magnitude
     smean = sum(s * delta) / 3.
     ds = s - smean * delta
     dsmag = sqrt(sum(w * ds * ds))

     ! yield stress
     y = y0
     if(c .ne. 0.d0) y = y + c * eqps ** (1 / m)
     radius = root23 * y

     ! check yield
     facyld = 0.d0
     if(dsmag - radius .ge. 0.d0) facyld = 1.d0
     dsmag = dsmag + (1.d0 - facyld)

     ! increment in plastic strain
     nddp = twomu
     h_kin = a
     h_iso = 0
     if(c .ne. 0.d0) h_iso = root23 * m * c * ((y - y0) / c) ** ((m-1)/m)
     dnom = nddp + h_kin + h_iso

     diff = dsmag - radius
     dlam = facyld * diff / dnom

     ! equivalet plastic strain
     eqps = eqps + root23 * dlam

     ! work with unit tensors
     dlam = dlam / dsmag

     ! update back stress
     fac = a * dlam
     bstress = bstress + fac * ds

     ! update stress
     fac = twomu * dlam
     sig = sig - fac * ds

     ! store data
     sigarg(1:6, ic) = sig
     svarg(1, ic) = eqps
     svarg(2:7, ic) = bstress

  end do gather_scatter

  return

end subroutine plast_calc
