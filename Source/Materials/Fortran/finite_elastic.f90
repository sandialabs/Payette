
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
  double precision :: c11, c12, c44, jac
  double precision, dimension(6) :: e, pk2, sig
  double precision, dimension(9) :: f

  ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ finite_elast_chk

  ! user properties
  c11 = ui(1)
  c12 = ui(2)
  c44 = ui(3)

  gather_scatter: do ic = 1, nc

     ! pass passed args to local
     f(1:9) = farg(1:9, ic)

     ! jacobian
     jac = f(1) * f(5) * f(9) + f(2) * f(6) * f(7) + f(3) * f(4) * f(8) &
         -(f(1) * f(6) * f(8) + f(2) * f(4) * f(9) + f(3) * f(5) * f(7))

     ! green lagrange strain
     e(1) = f(1) * f(1) + f(4) * f(4) + f(7) * f(7)
     e(2) = f(2) * f(2) + f(5) * f(5) + f(8) * f(8)
     e(3) = f(3) * f(3) + f(6) * f(6) + f(9) * f(9)
     e(4) = f(1) * f(2) + f(4) * f(5) + f(7) * f(8)
     e(5) = f(2) * f(3) + f(5) * f(6) + f(8) * f(9)
     e(6) = f(1) * f(3) + f(4) * f(6) + f(7) * f(9)
     e = .5d0 * (e - delta)

     ! pk2 stress
     pk2(1) = c11 * e(1) + c12 * e(2) + c12 * e(3)
     pk2(2) = c12 * e(1) + c11 * e(2) + c12 * e(3)
     pk2(3) = c12 * e(1) + c12 * e(2) + c11 * e(3)
     pk2(4:) = c44 * e(4:)

     ! cauchy stress
     sig(1) = f(1) * (f(1) * pk2(1) + f(2) * pk2(4) + f(3) * pk2(6)) + &
              f(2) * (f(1) * pk2(4) + f(2) * pk2(2) + f(3) * pk2(5)) + &
              f(3) * (f(1) * pk2(6) + f(2) * pk2(5) + f(3) * pk2(3))

     sig(2) = f(4) * (f(4) * pk2(1) + f(5) * pk2(4) + f(6) * pk2(6)) + &
              f(5) * (f(4) * pk2(4) + f(5) * pk2(2) + f(6) * pk2(5)) + &
              f(6) * (f(4) * pk2(6) + f(5) * pk2(5) + f(6) * pk2(3))

     sig(3) = f(7) * (f(7) * pk2(1) + f(8) * pk2(4) + f(9) * pk2(6)) + &
              f(8) * (f(7) * pk2(4) + f(8) * pk2(2) + f(9) * pk2(5)) + &
              f(9) * (f(7) * pk2(6) + f(8) * pk2(5) + f(9) * pk2(3))

     sig(4) = f(1) * (f(4) * pk2(1) + f(5) * pk2(4) + f(6) * pk2(6)) + &
              f(2) * (f(4) * pk2(4) + f(5) * pk2(2) + f(6) * pk2(5)) + &
              f(3) * (f(4) * pk2(6) + f(5) * pk2(5) + f(6) * pk2(3))

     sig(5) = f(4) * (f(7) * pk2(1) + f(8) * pk2(4) + f(9) * pk2(6)) + &
              f(5) * (f(7) * pk2(4) + f(8) * pk2(2) + f(9) * pk2(5)) + &
              f(6) * (f(7) * pk2(6) + f(8) * pk2(5) + f(9) * pk2(3))

     sig(6) = f(1) * (f(7) * pk2(1) + f(8) * pk2(4) + f(9) * pk2(6)) + &
              f(2) * (f(7) * pk2(4) + f(8) * pk2(2) + f(9) * pk2(5)) + &
              f(3) * (f(7) * pk2(6) + f(8) * pk2(5) + f(9) * pk2(3))

     sig = (1. / jac) * sig

     ! pass local to passed args
     earg(1:6, ic) = e(1:6)
     pk2arg(1:6, ic) = pk2(1:6)
     sigarg(1:6, ic) = sig(1:6)

  end do gather_scatter

  return

end subroutine finite_elast_calc
