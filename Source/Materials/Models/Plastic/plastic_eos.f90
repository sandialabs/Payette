  subroutine check_eos(ui)
    implicit none
    integer, parameter :: dk = selected_real_kind(14)
    real(kind=dk), dimension(*) :: ui
    real(kind=dk) :: dum
    dum=ui(1)
    call bombed("eos moduli not enabled")
  end subroutine check_eos
  subroutine eos_moduli(ui, pres, enrg, tmpr, rho, k, g)
    implicit none
    integer, parameter :: dk = selected_real_kind(14)
    real(kind=dk), dimension(*) :: ui
    real(kind=dk), dimension(1) :: pres, enrg, tmpr, rho
    real(kind=dk) :: k, g, dum
    dum=ui(1)
    dum=pres(1)
    dum=enrg(1)
    dum=tmpr(1)
    dum=rho(1)
    dum=k(1)
    dum=g(1)
    call bombed("eos moduli not enabled")
  end subroutine eos_moduli
