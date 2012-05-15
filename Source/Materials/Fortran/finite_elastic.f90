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


subroutine finite_elast_chk(ui)
  !******************************************************************************
  !     REQUIRED MIG DATA CHECK ROUTINE
  !     Checks validity of user inputs
  !     Sets defaults for unspecified user input.
  !     Adjusts user input to be self-consistent.
  !
  !******************************************************************************

  implicit none

  !........................................................................passed
  double precision, dimension (*) :: ui
  !.........................................................................local
  double precision :: k, mu, nu, c11, c12, c44
  character*16 iam
  parameter(iam='finite_elast_chk' )

  ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ finite_elast_chk

  mu = ui(1)
  nu = ui(2)
  k = ui(3)
  if(k .le. 0.d0) &
       call faterr(iam, "Bulk modulus K must be positive")
  if(mu .le. 0.d0) &
       call faterr(iam, "Shear modulus MU must be positive")
  if(nu .lt. 0.d0) &
       call logmes("WARNING: negative Poisson's ratio")

  ! redefine the elastic constants
  c11 = k + 4. / 3. * mu
  c12 = k - 2. / 3. * mu
  c44 = mu

  ui(1) = c11
  ui(2) = c12
  ui(3) = c44

  return
end subroutine finite_elast_chk


subroutine finite_elast_calc(nc, ui, farg, earg, pk2arg, sigarg)
  !******************************************************************************
  !
  !     Description:
  !       Hooke's law elasticity
  !
  !******************************************************************************
  !
  !     input arguments
  !     ===============
  !      NBLK       int                   Number of blocks to be processed
  !      NINSV      int                   Number of internal state vars
  !      UI         dp,ar(nprop)          User inputs
  !      farg       dp,ar(9)              Deformation gradient
  !
  !     output arguments
  !     ================
  !      earg       dp,ar(6)              Green-Lagrange strain
  !      pk2arg     dp,ar(6)              Second Piola-Kirchhoff stress
  !      sigarg     dp,ar(6)              Cauchy stress
  !
  !******************************************************************************
  !
  !      stresss and strains
  !          11, 22, 33, 12, 23, 13
  !
  !******************************************************************************

  use tensors

  implicit none

  !....................................................................parameters
  double precision, parameter, dimension(6) :: delta = (/1.,1.,1.,0.,0.,0./)
  double precision, parameter, dimension(6) :: w = (/1.,1.,1.,2.,2.,2./)

  !........................................................................passed
  integer :: nc
  double precision, dimension(*) :: ui
  double precision, dimension(6, nc) :: earg, pk2arg, sigarg
  double precision, dimension(9, nc) :: farg
  !.........................................................................local
  integer :: ic
  double precision :: c11, c12, c44
  double precision, dimension(6) :: e, pk2, sig
  ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ finite_elast_chk

  ! user properties
  c11 = ui(1)
  c12 = ui(2)
  c44 = ui(3)

  gather_scatter: do ic = 1, nc

     ! green lagrange strain
     e = .5d0 * (ata(farg(1:9, ic)) - delta)

     ! pk2 stress
     pk2(1) = c11 * e(1) + c12 * e(2) + c12 * e(3)
     pk2(2) = c12 * e(1) + c11 * e(2) + c12 * e(3)
     pk2(3) = c12 * e(1) + c12 * e(2) + c11 * e(3)
     pk2(4:) = c44 * e(4:)

     ! cauchy stress
     sig = push(pk2, farg(1:9, ic))

     ! pass local to passed args
     earg(1:6, ic) = e(1:6)
     pk2arg(1:6, ic) = pk2(1:6)
     sigarg(1:6, ic) = sig(1:6)

  end do gather_scatter

  return

end subroutine finite_elast_calc

