module tensors

  private delta, w, diag9
  public ata, symleaf, dd66x6, push, pull, ddp, mag, dev, iso, matinv

  double precision, parameter, dimension(6) :: &
       delta = (/1.d0, 1.d0, 1.d0, 0.d0, 0.d0, 0.d0/), &
       w = (/1.d0, 1.d0, 1.d0, 2.d0, 2.d0, 2.d0/)
  integer, parameter, dimension(3) :: diag9 = (/1, 5, 9/)


contains

  function ata(a)
    !**************************************************************************
    ! Compute Transpose(a).a
    !
    ! Parameters
    ! ----------
    ! a: tensor a stored as 9x1 Voight array
    !
    ! Returns
    ! -------
    ! ata: Symmetric tensor defined by Transpose(a).a stored as 6x1 Voight
    ! array
    !**************************************************************************
    implicit none
    !....................................................................passed
    double precision, dimension(6) :: ata
    double precision, dimension(9) :: a
    !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ata
    ata(1) = a(1) * a(1) + a(4) * a(4) + a(7) * a(7)
    ata(2) = a(2) * a(2) + a(5) * a(5) + a(8) * a(8)
    ata(3) = a(3) * a(3) + a(6) * a(6) + a(9) * a(9)
    ata(4) = a(1) * a(2) + a(4) * a(5) + a(7) * a(8)
    ata(5) = a(2) * a(3) + a(5) * a(6) + a(8) * a(9)
    ata(6) = a(1) * a(3) + a(4) * a(6) + a(7) * a(9)
    return
  end function ata

  function symleaf(f)
    !**************************************************************************
    ! Compute the 6x6 Mandel matrix (with index mapping {11,22,33,12,23,31})
    ! that is the sym-leaf transformation of the input 3x3 matrix F.

    ! If A is any symmetric tensor, and if {A} is its 6x1 Mandel array, then
    ! the 6x1 Mandel array for the tensor B=F.A.Transpose[F] may be computed
    ! by
    !                   {B}=[FF]{A}
    !
    ! If F is a deformation F, then B is the "push" (spatial) transformation
    ! of the reference tensor A. If F is Inverse[F], then B is the "pull"
    ! (reference) transformation of the spatial tensor A, and therefore B
    ! would be Inverse[FF]{A}.

    ! If F is a rotation, then B is the rotation of A, and FF would be be a
    ! 6x6 orthogonal matrix, just as is F.
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
    implicit none
    !....................................................................passed
    double precision, dimension(3, 3) :: f
    double precision, dimension(6, 6) :: symleaf
    !.....................................................................local
    integer i, j
    !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~symleaf
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
    !**************************************************************************
    implicit none
    !....................................................................passed
    double precision, dimension(6) :: a, push
    double precision, dimension(9) :: f
    !.....................................................................local
    double precision :: detf
    double precision, dimension(3, 3) :: ff
    !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~push
    detf = f(1) * f(5) * f(9) + f(2) * f(6) * f(7) + f(3) * f(4) * f(8) &
         -(f(1) * f(6) * f(8) + f(2) * f(4) * f(9) + f(3) * f(5) * f(7))
    ff = reshape(f, shape(ff))
    push = 1.d0 / detf * dd66x6(1, symleaf(ff), a)
    return
  end function push

  function pull(a, f)
    !**************************************************************************
    ! Performs the "pull" transformation
    !
    !            detf * Inverse(f).a.Transpose(Inverse(f))
    !
    ! For example, if a is the Cauchy stress, then the pull transofrmation
    ! returns the second Piola-Kirchoff stress
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
    !**************************************************************************
    implicit none
    !....................................................................passed
    double precision, dimension(6) :: a, pull
    double precision, dimension(9) :: f
    !.....................................................................local
    double precision :: detf
    double precision, dimension(3, 3) :: ff
    !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~pull
    detf = f(1) * f(5) * f(9) + f(2) * f(6) * f(7) + f(3) * f(4) * f(8) &
         -(f(1) * f(6) * f(8) + f(2) * f(4) * f(9) + f(3) * f(5) * f(7))
    ff = reshape(f, shape(ff))
    pull = detf * dd66x6(1, matinv(symleaf(ff), 6), a)
    return
  end function pull

  function unrot(a, r)
    !**************************************************************************
    implicit none
    !....................................................................passed
    double precision, dimension(6) :: a, unrot
    double precision, dimension(9) :: r
    double precision, dimension(3, 3) :: rt
    !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~unrot
    rt = transpose(reshape(r, shape(rt)))
    unrot = dd66x6(1, symleaf(rt), a)
    return
  end function unrot

  function rot(a, r)
    !**************************************************************************
    implicit none
    !....................................................................passed
    double precision, dimension(6) :: a, rot
    double precision, dimension(9) :: r
    double precision, dimension(3, 3) :: rr
    !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~rot
    rr = reshape(r, shape(rr))
    rot = dd66x6(1, symleaf(rr), a)
    return
  end function rot

  function inv(a)
    !**************************************************************************
    ! compute the inverse of a.
    !
    ! Parameters
    ! ----------
    ! a: 3x3 matrix stored as 9x1 array
    !
    ! Returns
    ! -------
    ! inv: inverse of a stored as 9x1 array
    !**************************************************************************
    implicit none
    !....................................................................passed
    double precision :: det
    double precision, dimension(9) :: a, inv
    !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~rot
    det = a(1) * a(5) * a(9) + a(2) * a(6) * a(7) + a(3) * a(4) * a(8) &
        -(a(1) * a(6) * a(8) + a(2) * a(4) * a(9) + a(3) * a(5) * a(7))
    inv = (/a(5) * a(9) - a(8) * a(6), &
            a(8) * a(3) - a(2) * a(9), &
            a(2) * a(6) - a(5) * a(3), &
            a(6) * a(7) - a(9) * a(4), &
            a(9) * a(1) - a(3) * a(7), &
            a(3) * a(4) - a(6) * a(1), &
            a(4) * a(8) - a(7) * a(5), &
            a(7) * a(2) - a(1) * a(8), &
            a(1) * a(5) - a(4) * a(2)/)
    inv = inv / det
    return
  end function inv

  function dp9x6(a, b)
    !**************************************************************************
    ! compute the dot product a.b
    !
    ! Parameters
    ! ----------
    ! a: 3x3 matrix stored as 9x1 array
    ! b: 3x3 symmetric matrix stored as 6x1 array
    !
    ! Returns
    ! -------
    ! dp9x6: a.b stored as a 9x1 array
    !**************************************************************************
    implicit none
    !....................................................................passed
    double precision, dimension(9) :: a, dp9x6
    double precision, dimension(6) :: b
    !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~dp9x6
    dp9x6 = (/a(1) * b(1) + a(4) * b(4) + a(7) * b(6), &
              a(4) * b(2) + a(1) * b(4) + a(7) * b(5), &
              a(7) * b(3) + a(4) * b(5) + a(1) * b(6), &
              a(2) * b(1) + a(5) * b(4) + a(8) * b(6), &
              a(5) * b(2) + a(2) * b(4) + a(8) * b(5), &
              a(8) * b(3) + a(5) * b(5) + a(2) * b(6), &
              a(3) * b(1) + a(6) * b(4) + a(9) * b(6), &
              a(6) * b(2) + a(3) * b(4) + a(9) * b(5), &
              a(9) * b(3) + a(6) * b(5) + a(3) * b(6)/)
    return
  end function dp9x6

  function ddp(a, b)
    !**************************************************************************
    ! compute the double dot product a:b
    !
    ! Parameters
    ! ----------
    ! a: 3x3 symmetric matrix stored as 6x1 array
    ! b: 3x3 symmetric matrix stored as 6x1 array
    !
    ! Returns
    ! -------
    ! ddp: a:b stored as a 6x1 array
    !**************************************************************************
    implicit none
    !....................................................................passed
    double precision ddp
    double precision, dimension(6) :: a, b
    !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ddp
    ddp = sum(w * a * b)
    return
  end function ddp

  function dp6x3(a, b)
    !**************************************************************************
    ! compute the a.b
    !
    ! Parameters
    ! ----------
    ! a: symmetric 3x3 matrix stored as 6x1 array
    ! b: 3x1 array
    !
    ! Returns
    ! -------
    ! dp6x3: 3x1 array
    !**************************************************************************
    implicit none
    !....................................................................passed
    double precision, dimension(6) :: a
    double precision, dimension(3) :: b, dp6x3
    !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~dp6x3
    dp6x3 = (/a(1) * b(1) + a(4) * b(2) + a(6) * b(3), &
              a(4) * b(1) + a(2) * b(2) + a(5) * b(3), &
              a(6) * b(1) + a(5) * b(2) + a(3) * b(3)/)
    return
  end function dp6x3

  function dyad(a, b)
    !**************************************************************************
    ! compute the dyadic product axb
    !
    ! Parameters
    ! ----------
    ! a: 3x1 array
    ! b: 3x1 array
    !
    ! Returns
    ! -------
    ! dyad: 3x3 symmetric matrix stored as 6x1 array
    !**************************************************************************
    implicit none
    !....................................................................passed
    double precision, dimension(3) :: a, b
    double precision, dimension(6) :: dyad
    !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~dp6x3
    dyad = (/a(1) * b(1), a(2) * b(2), a(3) * b(3), &
             a(1) * b(2), a(2) * b(3), a(1) * b(3)/)
    return
  end function dyad

  function mag(a)
    !**************************************************************************
    ! compute the magnitude of a
    !
    ! Parameters
    ! ----------
    ! a: 3x3 symmetric matrix stored as 6x1 array
    !
    ! Returns
    ! -------
    ! mag: sqrt(a:a) stored as a 6x1 array
    !**************************************************************************
    implicit none
    !....................................................................passed
    double precision mag
    double precision, dimension(6) :: a
    !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~mag
    mag = sqrt(ddp(a, a))
    return
  end function mag

  function dev(a)
    !**************************************************************************
    ! compute the deviatoric part of a
    !
    ! Parameters
    ! ----------
    ! a: 3x3 symmetric matrix stored as 6x1 array
    !
    ! Returns
    ! -------
    ! dev: deviatoric part of a stored as a 6x1 array
    !**************************************************************************
    implicit none
    !....................................................................passed
    double precision, dimension(6) :: dev, a
    !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~dev
    dev = a - iso(a)
    return
  end function dev

  function iso(a)
    !**************************************************************************
    ! compute the isotropic part of a
    !
    ! Parameters
    ! ----------
    ! a: 3x3 symmetric matrix stored as 6x1 array
    !
    ! Returns
    ! -------
    ! iso: isotropic part of a stored as a 6x1 array
    !**************************************************************************
    implicit none
    !....................................................................passed
    double precision, dimension(6) :: iso, a
    !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~iso
    iso = ddp(a, delta) / 3. * delta
    return
  end function iso

  function matinv(a, n)
    !**************************************************************************
    ! This procedure computes the inverse of a real, general matrix using
    ! Gauss- Jordan elimination with partial pivoting. The input matrix, A, is
    ! returned unchanged, and the inverted matrix is returned in matinv. The
    ! procedure also returns an integer flag, icond, indicating whether A is
    ! well- or ill- conditioned. If the latter, the contents of matinv will be
    ! garbage.
    !
    ! The logical dimensions of the matrices A(1:n,1:n) and matinv(1:n,1:n)
    ! are assumed to be the same as the physical dimensions of the storage
    ! arrays a(1:np,1:np) and matinv(1:np,1:np), i.e., n = np. If A is not
    ! needed, a and matinv can share the same storage locations.
    !
    ! Parameters
    ! ----------
    ! A: real  matrix to be inverted
    ! n: int   number of rows/columns
    !
    ! Returns
    ! -------
    ! matinv   real  inverse of A
    !**************************************************************************
    implicit none
    !....................................................................passed
    integer :: n
    double precision, dimension(n, n) :: a, matinv
    !.....................................................................local
    integer :: row, col, icond
    integer, dimension(1) :: v
    double precision :: wmax, fac, wcond=1.d-13
    double precision, dimension(n) :: dum
    double precision, dimension(n, n) :: w
    !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~matinv

    ! Initialize

    icond = 0.d0
    matinv  = 0.d0
    do row = 1, n
       matinv(row, row) = 1.d0
    end do
    w = a
    do row = 1, n
       v = maxloc(abs(w(row,:)))
       wmax = w(row, v(1))
       if (Wmax == 0) then
          icond = 1
          return
       end if
       w(row, :) = w(row, :) / wmax
       matinv(row, :) = matinv(row, :) / wmax
    end do

    ! Gauss-Jordan elimination with partial pivoting

    do col = 1, n
       v = maxloc(abs(w(col:, col)))
       row = v(1) + col - 1
       dum(col:) = w(col, col:)
       w(col, col:) = w(row, col:)
       w(row, col:) = dum(col:)
       dum(:) = matinv(col, :)
       matinv(col, :) = matinv(row, :)
       matinv(row, :) = dum(:)
       wmax = w(col, col)
       if(abs(wmax) .lt. wcond) then
          icond = 1
          return
       end if
       row = col
       w(row,col:) = w(row, col:) / wmax
       matinv(row, :) = matinv(row, :) / wmax
       do row = 1, n
          if(row .eq. col) cycle
          fac = w(row, col)
          w(row, col:) = w(row, col:) - fac * w(col, col:)
          matinv(row, :) = matinv(row, :) - fac * matinv(col, :)
       end do
    end do

    return

  end function matinv

end module tensors


