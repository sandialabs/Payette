module futalafu_constants
  integer, parameter :: dk=selected_real_kind(14)
  integer, parameter :: ik=selected_int_kind(4)

  ! parameter pointers
  integer(kind=ik), parameter :: nrui=11, nfui=5, neos=6
  integer(kind=ik), parameter :: nui=nrui+nfui+neos
  integer(kind=ik), parameter :: ipk=1
  integer(kind=ik), parameter :: ipmu=2
  integer(kind=ik), parameter :: ipnu=3
  integer(kind=ik), parameter :: ipy0=4
  integer(kind=ik), parameter :: ipa=5
  integer(kind=ik), parameter :: ipc0=6
  integer(kind=ik), parameter :: ipc1=7
  integer(kind=ik), parameter :: ipc2=8
  integer(kind=ik), parameter :: ipyf=9
  integer(kind=ik), parameter :: ipgf=10
  integer(kind=ik), parameter :: ipsf=11

  ! free parameters
  integer(kind=ik), parameter :: ipfree=nrui
  integer(kind=ik), parameter :: ipf05=ipfree+1
  integer(kind=ik), parameter :: ipf04=ipfree+2
  integer(kind=ik), parameter :: ipf03=ipfree+3
  integer(kind=ik), parameter :: ipf02=ipfree+4
  integer(kind=ik), parameter :: ipf01=nrui+nfui

  ! eos parameters -> input by user
  integer(kind=ik), parameter :: ipeos=nrui+nfui
  integer(kind=ik), parameter :: ipeosid=ipeos+1
  integer(kind=ik), parameter :: ipr0=ipeos+2
  integer(kind=ik), parameter :: ipt0=ipeos+3
  integer(kind=ik), parameter :: ipcs=ipeos+4
  integer(kind=ik), parameter :: ips1=ipeos+5
  integer(kind=ik), parameter :: ipgp=ipeos+6
  integer(kind=ik), parameter :: ipcv=nrui+nfui+neos

  ! state variable pointers
  integer(kind=ik), parameter :: nrxtra=15, nfxtra=5,nxtra=nrxtra+nfxtra
  integer(kind=ik), parameter :: kgam=1
  integer(kind=ik), parameter :: kepv=2
  integer(kind=ik), parameter :: kdmg=3
  integer(kind=ik), parameter :: kbs=3
  integer(kind=ik), parameter :: kbsxx=kbs+1 ! 4
  integer(kind=ik), parameter :: kbsyy=kbs+2 ! 5
  integer(kind=ik), parameter :: kbszz=kbs+3 ! 6
  integer(kind=ik), parameter :: kbsxy=kbs+4 ! 7
  integer(kind=ik), parameter :: kbsyz=kbs+5 ! 8
  integer(kind=ik), parameter :: kbsxz=kbs+6 ! 9
  integer(kind=ik), parameter :: ktherm=kbsxz
  integer(kind=ik), parameter :: kenrgy=ktherm+1 ! 10
  integer(kind=ik), parameter :: ktmpr=ktherm+2 ! 11
  integer(kind=ik), parameter :: krho=ktherm+3 ! 12
  integer(kind=ik), parameter :: kmech=ktherm+3 ! 13
  integer(kind=ik), parameter :: kr=kmech+1 ! 14
  integer(kind=ik), parameter :: kz=kmech+2 ! 15
  integer(kind=ik), parameter :: kyld=nrxtra ! 16
  integer(kind=ik), parameter :: kfree=kyld
  integer(kind=ik), parameter :: kf01=kfree+1 ! 17
  integer(kind=ik), parameter :: kf02=kfree+2 ! 18
  integer(kind=ik), parameter :: kf03=kfree+3 ! 19
  integer(kind=ik), parameter :: kf04=kfree+4 ! 20
  integer(kind=ik), parameter :: kf05=nxtra ! 21

  ! numbers
  real(kind=dk), parameter :: half=.5_dk
  real(kind=dk), parameter :: zero=0._dk, one=1._dk, two=2._dk
  real(kind=dk), parameter :: three=3._dk, six=6._dk, ten=10._dk

end module futalafu_constants


