module tensors

contains

  function tada(a)
    !**************************************************************************
    ! Compute Transpose(a).a
    !
    ! Parameters
    ! ----------
    ! a: tensor a stored as 9x1 Voight array
    !
    ! Returns
    ! -------
    ! tada: Symmetric tensor defined by Transpose(a).a stored as 6x1 Voight
    ! array
    !**************************************************************************
    implicit none
    !....................................................................passed
    double precision, dimension(6) :: tada
    double precision, dimension(9) :: a
    !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~tada
    tada(1) = a(1) * a(1) + a(4) * a(4) + a(7) * a(7)
    tada(2) = a(2) * a(2) + a(5) * a(5) + a(8) * a(8)
    tada(3) = a(3) * a(3) + a(6) * a(6) + a(9) * a(9)
    tada(4) = a(1) * a(2) + a(4) * a(5) + a(7) * a(8)
    tada(5) = a(2) * a(3) + a(5) * a(6) + a(8) * a(9)
    tada(6) = a(1) * a(3) + a(4) * a(6) + a(7) * a(9)
    return
  end function tada

  !****************************************************************************

  function symleaf(farg)
    !**************************************************************************
    ! Compute the 6x6 Mandel matrix (with index mapping {11,22,33,12,23,31})
    ! that is the sym-leaf transformation of the input 3x3 matrix F.

    ! If A is any symmetric tensor, and if {A} is its 6x1 Mandel array, then
    ! the 6x1 Mandel array for the tensor B=F.A.Transpose[F] may be computed
    ! by
    !                   {B}=[FF]{A}
    !
    ! If F is a deformation F, then B is the "push" (spatial) transformation
    ! of the reference tensor A If F is Inverse[F], then B is the "pull"
    ! (reference) transformation of the spatial tensor A, and therefore B
    ! would be Inverse[FF]{A}.

    ! If F is a rotation, then B is the rotation of A, and FF would be be a
    ! 6x6 orthogonal matrix, just as is F
    !
    ! Parameters
    ! ----------
    ! f: any matrix (in conventional 3x3 storage)
    !
    ! Returns
    ! -------
    ! symleaf: 6x6 Mandel matrix for the sym-leaf transformation matrix
    !
    !
    ! authors
    ! -------
    !  rmb:Rebecca Brannon:theory, algorithm, and code
    !
    ! modification history
    ! --------------------
    !  yymmdd|who|what was done
    !  ------ --- -------------
    !  060915|rmbrann|created routine
    !  120514|tjfulle|conversion to f90
    !**************************************************************************
    !....................................................................passed
    double precision, dimension(9) :: farg
    double precision, dimension(6, 6) :: symleaf
    !.....................................................................local
    integer i, j
    double precision, dimension(3, 3) :: f
    !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~symleaf
    f = reshape(farg, shape(f))
    do i = 1, 3
       do j = 1, 3
          symleaf(i, j) = f(i, j) ** 2
       end do
       symleaf(i, 4) = sqrt(2.d0) * f(i, 1) * f(i, 2)
       symleaf(i, 5) = sqrt(2.d0) * f(i, 2) * f(i, 3)
       symleaf(i, 6) = sqrt(2.d0) * f(i, 3) * f(i, 1)
       symleaf(4, i) = sqrt(2.d0) * f(1, i) * f(2, i)
       symleaf(5, i) = sqrt(2.d0) * f(2, i) * f(3, i)
       symleaf(6, i) = sqrt(2.d0) * f(3, i) * f(1, i)
    enddo
    symleaf(4, 4) = f(1, 2) * f(2, 1) + f(1, 1) * f(2, 2)
    symleaf(5, 4) = f(2, 2) * f(3, 1) + f(2, 1) * f(3, 2)
    symleaf(6, 4) = f(3, 2) * f(1, 1) + f(3, 1) * f(1, 2)

    symleaf(4, 5) = f(1, 3) * f(2, 2) + f(1, 2) * f(2, 3)
    symleaf(5, 5) = f(2, 3) * f(3, 2) + f(2, 2) * f(3, 3)
    symleaf(6, 5) = f(3, 3) * f(1, 2) + f(3, 2) * f(1, 3)

    symleaf(4, 6) = f(1, 1) * f(2, 3) + f(1, 3) * f(2, 1)
    symleaf(5, 6) = f(2, 1) * f(3, 3) + f(2, 3) * f(3, 1)
    symleaf(6, 6) = f(3, 1) * f(1, 3) + f(3, 3) * f(1, 1)
    return
  end function symleaf


  !****************************************************************************


  function dd66x6(job, a, x)
    !**************************************************************************
    ! Compute the product of a fourth-order tensor a times a second-order
    ! tensor x (or vice versa if job=-1)
    !              a:x if job=1
    !              x:a if job=-1
    !
    ! Parameters
    ! ----------
    !    a: 6x6 Mandel matrix for a general (not necessarily major-sym)
    !       fourth-order minor-sym matrix
    !    x: 6x6 Voigt matrix
    !
    ! returns
    ! -------
    !    a:x if JOB=1
    !    x:a if JOB=-1
    !
    ! authors
    ! -------
    !  rmb:Rebecca Brannon:theory, algorithm, and code
    !
    ! modification history
    ! --------------------
    !  yymmdd|who|what was done
    !  ------ --- -------------
    !  060915|rmb|created routine
    !  120514|tjfulle|conversion to f90
    !**************************************************************************
    implicit none
    !....................................................................passed
    integer job
    double precision, dimension(6) :: x, dd66x6
    double precision, dimension(6, 6) :: a
    !.....................................................................local
    double precision, parameter, dimension(6) :: &
         mandel=(/1.d0,1.d0,1.d0,sqrt(2.d0),sqrt(2.d0),sqrt(2.d0)/)
    double precision, dimension(6) :: t
    !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~dd66x6

    !  Construct the Mandel version of x
    t = mandel * x

    select case(job)
    case(1)
       ! Compute the Mandel form of A:X
       dd66x6 = matmul(a, t)

    case(-1)
       ! Compute the Mandel form of X:A
       dd66x6 = matmul(t, a)

    case default
       call bombed('unknown job sent to dd66x6')

    end select

    ! Convert result to Voigt form
    dd66x6 = dd66x6 / mandel

    return
  end function dd66x6


  !****************************************************************************


  function push(a, f)
    !**************************************************************************
    ! Performs the "push" transformation
    !
    !             1
    !            ---- f.a.Transpose(f)
    !            detf
    !
    ! For example, if a is the Second-Piola Kirchoff stress, then the push
    ! transofrmation returns the Cauchy stress
    !
    ! Parameters
    ! ----------
    !    x: 6x1 Voigt array
    !    f: 9x1 deformation gradient, stored as Voight array
    !
    ! returns
    ! -------
    ! push: 6x1 Voight array
    !
    ! modification history
    ! --------------------
    !  yymmdd|who|what was done
    !  ------ --- -------------
    !  120514|tjfulle|created subroutine
    !....................................................................passed
    double precision, dimension(6) :: a, push
    double precision, dimension(9) :: f
    !.....................................................................local
    double precision :: detf
    !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~push
    detf = f(1) * f(5) * f(9) + f(2) * f(6) * f(7) + f(3) * f(4) * f(8) &
         -(f(1) * f(6) * f(8) + f(2) * f(4) * f(9) + f(3) * f(5) * f(7))
    push = dd66x6(1, symleaf(f), a) / detf
    return
  end function push

end module tensors


