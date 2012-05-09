
subroutine elast_chk(ui)
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
  double precision :: k, mu, nu
  character*9 iam
  parameter(iam='elast_chk' )

  ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ elast_chk

  k = ui(1)
  mu = ui(2)
  if(k .le. 0.d0) &
       call faterr(iam, "Bulk modulus K must be positive")
  if(mu .le. 0.d0) &
       call faterr(iam, "Shear modulus MU must be positive")

  ! poisson's ratio
  nu = (3.d0 * k - 2.d0 * mu) / (6.d0 * k + 2.d0 * mu)
  if(nu .lt. 0.d0) &
       call logmes("WARNING: negative Poisson's ratio")

  return
end subroutine elast_chk


subroutine elast_calc(nc, dt, ui, sigarg, darg)
  !***********************************************************************
  !
  !     Description:
  !       Hooke's law elasticity
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
  !
  !     output arguments
  !     ================
  !      USM      dp                      uniaxial strain modulus
  !
  !***********************************************************************
  !
  !      stresss and strains
  !          11, 22, 33, 12, 23, 13
  !
  !***********************************************************************

  implicit none

  !.................................................................. parameters
  double precision, parameter, dimension(6) :: delta = (/1.,1.,1.,0.,0.,0./)
  double precision, parameter, dimension(6) :: w = (/1.,1.,1.,2.,2.,2./)

  !...................................................................... passed
  integer :: nc
  double precision :: dt
  double precision, dimension(*) :: ui
  double precision, dimension(6, nc) :: sigarg, darg
  !...................................................................... local
  integer :: ic
  double precision :: k, mu, twomu, alam, trde
  double precision, dimension(6) :: de

  ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ elast_chk

  ! user properties
  k = ui(1)
  mu = ui(2)

  ! constants
  twomu = 2. * mu
  alam = k - twomu / 3.

  gather_scatter: do ic = 1, nc

     ! get passed arguments
     de = darg(1:6, ic) * dt

     ! elastic stress update
     trde = sum(de * delta)
     sigarg(1:6, ic) = sigarg(1:6, ic) + alam * trde * delta + twomu * de

  end do gather_scatter

  return

end subroutine elast_calc
