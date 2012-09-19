  subroutine check_eos(ui)
    implicit none
    integer, parameter :: dk = selected_real_kind(14)
    real(kind=dk), dimension(*) :: ui
    call bombed("eos moduli not enabled")
  end subroutine check_eos
  subroutine get_eos_moduli(ui, pres, enrg, tmpr, rho, k, g)
    implicit none
    integer, parameter :: dk = selected_real_kind(14)
    real(kind=dk), dimension(*) :: ui
    real(kind=dk), dimension*(1) :: pres, enrg, tmpr, rho
    real(kind=dk) :: k, g
    call bombed("eos moduli not enabled")
  end subroutine get_eos_moduli
