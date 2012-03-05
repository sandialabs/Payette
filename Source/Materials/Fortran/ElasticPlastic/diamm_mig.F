c The MIT License
c
c Copyright (c) 2011 Tim Fuller
c
c License for the specific language governing rights and limitations under
c Permission is hereby granted, free of charge, to any person obtaining a
c copy of this software and associated documentation files (the "Software"),
c to deal in the Software without restriction, including without limitation
c the rights to use, copy, modify, merge, publish, distribute, sublicense,
c and/or sell copies of the Software, and to permit persons to whom the
c Software is furnished to do so, subject to the following conditions:
c
c The above copyright notice and this permission notice shall be included
c in all copies or substantial portions of the Software.
c
c THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
c OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
c FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
c THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
c LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
c FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
c DEALINGS IN THE SOFTWARE.

C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      SUBROUTINE DMMCHK(UI,GC,DC)
C
C***********************************************************************
C     REQUIRED MIG DATA CHECK ROUTINE
C     Checks validity of user inputs for DMM model.
C     Sets defaults for unspecified user input.
C     Adjusts user input to be self-consistent.
C
C     input
C     -----
C       UI: user input as read and stored by host code.
C
C       Upon entry, the UI array contains the user inputs EXACTLY
C       as read from the user.  These inputs must be ordered in the
C       UI array as indicated in the file kmmpnt.Blk.
C       See kmmpnt.Blk for parameter definitions and keywords.
C
C       DC: Not used with this model
C
C    Other output
C    ------------
C       GC: Not used with this model
C       DC: Not used with this model
C       Because GC and DC are not used, you may call this routine
C       with a line of the form "CALL DMMCHK(UI,UI,UI)"
C
C***********************************************************************
C  author:  Rebecca Brannon
C
C  yymmdd:usernam:   m o d i f i c a t i o n
C  ---------------------------------------------------------------------
C  090713:tim fuller:Created original data check
C
#include "precision.Blk"
#include "diamm_pnt.Blk"
C
      DIMENSION UI(*), GC(*), DC(*)
      PARAMETER (PZERO=0.D0,PONE=1.D0,PTWO=2.D0,PTHREE=3.D0)
      PARAMETER (HUGE=1.D80)
      PARAMETER (  ROOT3=0.173205080756887729352744634150587236694281D1)
      PARAMETER (  TOOR3=PONE/ROOT3)
C
C  ...local
      CHARACTER*6 IAM
      PARAMETER( IAM = 'DMMCHK' )
      LOGICAL DEJAVU
      DATA DEJAVU/.FALSE./

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
c     avoid compiler warning of unused dummy arguments
      dum=gc(1)
      dum=dc(1)
c
      DEJAVU=(NINT(UI(IPDEJAVU)).NE.0)
C
      IF(.NOT.DEJAVU)THEN
        CALL LOGMES('############# DMM build 100222')
      ENDIF
C
C     Check validity of user-supplied model parameters
C
      IF(UI(IPRHO0).LT.PZERO)CALL FATERR(IAM,'R0 must be positive')
      IF(UI(IPB0).LE.PZERO)THEN
         IF(UI(IPC0).LE.PZERO)THEN
            CALL FATERR(IAM,'B0 or C0 must be positive')
         ELSE
            UI(IPB0)=UI(IPRHO0)*UI(IPC0)*UI(IPC0)
         ENDIF
      ELSE
         IF(UI(IPC0).LE.PZERO)THEN
            UI(IPC0)=SQRT(UI(IPB0)/UI(IPRHO0))
         ENDIF
      ENDIF
      IF(UI(IPG0).LE.PZERO)CALL FATERR(IAM,'G0 must be positive')
      IF(UI(IPA1).LT.PZERO)CALL FATERR(IAM,'A1 must be nonnegative')
      IF(UI(IPA2).LT.PZERO)CALL FATERR(IAM,'A2 must be nonnegative')
      IF(UI(IPA3).LT.PZERO)CALL FATERR(IAM,'A3 must be nonnegative')
      IF(UI(IPA4).LT.PZERO)CALL FATERR(IAM,'A4 must be nonnegative')
      IF(UI(IPA4PF).LT.PZERO)CALL FATERR(IAM,'A4PF must be nonnegative')

c     Set defaults
      IF(UI(IPA4PF).EQ.PZERO)UI(IPA4PF)=UI(IPA4)  !default A4PF = A4

c     Convert derivative of G w.r.t. pressure to derivative w.r.t.
c     elastic volume change
      IF(.NOT.DEJAVU)THEN
         IF(UI(IPG1).LE.PZERO)THEN
            UI(IPG1)=PZERO
            UI(IPAN)=PZERO
         ENDIF
         IF(UI(IPB1).LE.PZERO)UI(IPB1)=PZERO
         UI(IPB1)= -UI(IPB1)*UI(IPB0)
         UI(IPG1)= -UI(IPG1)*UI(IPB0)
      ENDIF

C     Get Drucker Prager coefficients if compressive strength is given
      IF(UI(IPA1).LT.HUGE.AND.UI(IPSC).GT.PZERO)THEN
         IF(UI(IPA4).LE.PZERO)THEN
            UI(IPA4)=TOOR3*(UI(IPSC)-UI(IPA1))/(UI(IPSC)+UI(IPA1))
c     UI(IPA1)=PTWO/ROOT3*UI(IPSC)*UI(IPA1)/(UI(IPSC)+UI(IPA1))
         ENDIF
      ENDIF
      IF(UI(IPT0).LE.PZERO)UI(IPT0)=298.D0
      IF(UI(IPRHO0).LE.PZERO)UI(IPRHO0)=PONE
      IF(UI(IPTM).LE.PZERO)UI(IPTM)=1.D99
      IF(UI(IPCV).LE.PZERO)UI(IPCV)=PONE
      IF(UI(IPGP).LE.PZERO)UI(IPGP)=PZERO
      IF(UI(IPXP).LE.PZERO)UI(IPXP)=PONE
      IF(UI(IPTQC).LE.PZERO)UI(IPTQC)=PONE

      IF(.NOT.DEJAVU)THEN
         IF(UI(IPA4PF).EQ.UI(IPA4))THEN
            CALL LOGMES(IAM//': FYI, This material is associative')
         ELSE
            CALL LOGMES(IAM//': FYI, This material is nonassociative')
         ENDIF
        CALL LOGMES('############# DMM data check complete')
        UI(IPDEJAVU)=PONE
        IF(PTHREE*UI(IPB0).LT.PTWO*UI(IPG0))THEN
           CALL LOGMES
     & (IAM//' Warning: neg Poisson (to avoid warning, set 3*B0>2*G0)')
        ENDIF
      ENDIF
      RETURN
      END !SUBROUTINE DMMCHK
C
C
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      SUBROUTINE DMMRXV(UI,GC,DC,
     &  NX, NAMEA, KEYA, RINIT, RDIM, IADVCT, ITYPE)
C**********************************************************************
C     REQUESTED EXTRA VARIABLES FOR KAYENTA
C
C     This subroutine creates lists of the internal state variables
C     needed for DMM. This routine merely sends a
C     LIST of internal state variable requirements back to the host
C     code.   IT IS THE RESPONSIBILITY OF THE HOST CODE to loop over
C     the items in each list to actually establish necessary storage
C     and (if desired) set up plotting, restart, and advection
C     control for each internal state variable.
C
C     called by: host code after all input data have been checked
C
C     input
C     -----
C          UI = user input array
C          GC = unused for this model (placeholder)
C          DC = unused for this model (placeholder)
C
C     output
C     ------
C          NX = number of extra variables                    [DEFAULT=0]
C       NAMEA = single character array created from a string array
C               (called NAME) used locally in this routine to register
C               a descriptive name for each internal state variable.
C        KEYA = single character array created from a string array
C               (called KEY) used locally in this routine to register
C               a plot keyword for each internal state variable.
C          | Note: NAMEA and KEYA are created from the local variables |
C          | NAME and KEY by calls to the subroutine TOKENS, which     |
C          | is a SERVICE routine presumed to ALREADY exist within the |
C          | host code (we can provide this routine upon request).     |
C          | "NAME" is a fortran array of strings. "NAMEA" is a one    |
C          | dimensional array of single characters. For readability,  |
C          | most of this subroutine writes to the NAME array. Only at |
C          | the very end is NAME converted to NAMEA by calling a      |
C          | the utility routine TOKENS. The KEY array is similarly    |
C          | converted to KEYA.  These conversions are performed       |
C          | because host codes written in C or C++ are unable to      |
C          | process FORTRAN string arrays. Upon request, we can       |
C          | provide a utility routine that will convert BACK to       |
C          | FORTRAN string arrays if your host code is FORTRAN.       |
C          | Likewise, we can provide a C++ routine that will allow    |
C          | parsing the single-character arrays to convert them back  |
C          | to strings if your code is C++. Alternatively, you can    |
C          | simply ignore the NAMEA and KEYA outputs of this routine  |
C          | if your host code does not wish to establish plotting     |
C          | information.                                              |
C
C       RINIT = initial value for each ISV               [DEFAULT = 0.0]
C        RDIM = physical dimension exponents             [DEFAULT = 0.0]
C               This variable is dimensioned RDIM(7,*) for the 7 base
C               dimensions (and * for the number of extra variables):
C
C                      1 --- length
C                      2 --- mass
C                      3 --- time
C                      4 --- temperature
C                      5 --- discrete count
C                      6 --- electric current
C                      7 --- luminous intensity
C
C                Suppose, for example, that an ISV has units of stress.
C                Dimensionally, stress is length^(1) times mass^(-1)
C                times time^(-2). Therefore, this routine would return
C                1.0, -1.0, and -2.0 as the first three values of the
C                RDIM array. Host codes that work only in one unit
C                set (e.g., SI) typically ignore the RDIM output.
C
C      IADVCT = advection option                           [DEFAULT = 0]
C                    = 0 advect by mass-weighted average
C                    = 1 advect by volume-weighted average
C                    = 2 don't advect
C            The advection method will often be ignored by host codes.
C            It is used for Eulerian codes and for Lagrangian codes that
C            re-mesh (and therefore need guidance about how to "mix"
C            internal state variables). Note: a value of 2 implies that
C            the ISV is output only.
C
C        ITYPE = variable type (see migtionary preface)    [DEFAULT = 1]
C                  1=scalar
C                  6=2nd-order symmetric tensor
C        The component ordering for ITYPE=6 is 11, 22, 33, 12, 23, 31.
C        Consequently, the 11 component is the first one to be requested
C        in tensor lists, and its IFLAG is set to 6. To indicate that
C        subsequent ISVs are the remaining components of the same tensor,
C        the next five ISVs are given an IFLAG value of -6.
C        Host codes that don't change basis can ignore ITYPE.
C
C***********************************************************************
C
C  author:  Rebecca Brannon
C
C    who    yymmdd  M O D I F I C A T I O N
C  -------  ------  ----------------------------------------------------
C  rmbrann  030809  Created original extra variable routine
C
      IMPLICIT NONE
#include "diamm_pnt.Blk"
C
      INTEGER MMCN,MMCK,MNUNIT,MMCNA,MMCKA
      DOUBLE PRECISION PZERO,PONE,PTWO,PTHREE
      PARAMETER(MMCN=50,MMCK=10,MNUNIT=7)
      PARAMETER(MMCNA=NDMMISV*MMCN,MMCKA=NDMMISV*MMCK)
      PARAMETER(PZERO=0.D0,PONE=1.D0,PTWO=2.D0,PTHREE=3.D0)

C
      INTEGER NX,IADVCT,ITYPE
      DOUBLE PRECISION UI,GC,DC,RINIT,RDIM
      CHARACTER*(MMCN) NAME(NDMMISV)
      CHARACTER*(MMCK) KEY(NDMMISV)
      CHARACTER*1 NAMEA(*), KEYA(*)
      DIMENSION IADVCT(*),ITYPE(*)
      DIMENSION UI(*), GC(*), DC(*), RINIT(*), RDIM(7,*)

      INTEGER ISV
      DOUBLE PRECISION VINIT,dum
      LOGICAL SERIAL
      DATA SERIAL/.FALSE./
C
      CHARACTER*6 IAM
      PARAMETER(IAM='DMMRXV')

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC DMMRXV
*     These next lines are only to avoid compiler warnings
      dum=dc(1)
      dum=gc(1)
      SERIAL= UI(IPTEST).NE.0
      VINIT=0.D0
#ifdef TJF_MED
      IF(SERIAL)VINIT=1.D0
#endif
*
      CALL LOGMES('############# Requesting DMM variables')
C
C
C
C
C     ***********************************************************
C     ***   SET INITIAL VALUES FOR STATE VARIABLES            ***
C     ***                                                     ***
C     ***   The first argument is the user input (previously  ***
C     ***   obtained by reading an input set). The next       ***
C     ***   argument, RINIT, is the list of initial values    ***
C     ***   for each of the internal state variables listed   ***
C     ***   below.                                            ***
C     ***                                                     ***
C     ***********************************************************
C
C
C     For now, the ISVs will be initialized using an assumption
C     that the initial stress is zero.  Host codes that support a
C     nonzero initial stress field will need to call
C     ISOTROPIC_DMMMATERIAL_INIT with the
C     initial stress in each element to reset the
C     ISVs appropriately.
C
      DO ISV=1,NDMMISV
         RINIT(ISV)=PZERO
      ENDDO
C
C
      NX=0
C
C----------------------------------------------------------------------1
      NX=NX+1                                                    !KEQDOT
      IF(NX.NE.KEQDOT)CALL BOMBED(IAM//' KEQDOT pointer wrong')
      NAME(NX) = 'Magnitude of the total strain rate'
      KEY(NX) = 'EQDOT'
      IADVCT(NX)=2        ! output only
C     itype=1  (scalar)
      RDIM(1,NX)=  PZERO  ! units: 1/time
      RDIM(2,NX)=  PZERO
      RDIM(3,NX)= -PONE
      RINIT(NX)=VINIT
C----------------------------------------------------------------------2
      NX=NX+1                                                       !kI1
      IF(NX.NE.KI1)CALL BOMBED(IAM//' KI1 pointer wrong')
      NAME(NX)='First invariant (trace) of stress'
C     Note: stress is positive in tension. Therefore, I1 is typically
C     negative in compression. The mechanical pressure (positive in
C     compression) and mean stress (positive in tension) are given by
C           pressure    = - I1/3
C           mean stress = + I1/3
      KEY(NX)='I1'
      IADVCT(NX)=2        ! output only
C     itype=1  (scalar)
      RDIM(1,NX)= -PONE   ! units: stress
      RDIM(2,NX)=  PONE
      RDIM(3,NX)= -PTWO
      RINIT(NX)=VINIT
C----------------------------------------------------------------------3
      NX=NX+1                                                   !KROOTJ2
      IF(NX.NE.KROOTJ2)CALL BOMBED(IAM//' KROOTJ2 pointer wrong')
      NAME(NX)='Square root of second stress invariant, SQRT(J2)'
C     The second stress invariant is given by
C            J2 = (1/2) trace(S.S)
C     where S is the stress deviator. Therefore,
C       ROOTJ2 = magnitude(S)/sqrt(2)
      KEY(NX)='ROOTJ2'
      IADVCT(NX)=2        ! output only
C     itype=1  (scalar)
      RDIM(1,NX)= -PONE   ! units: stress
      RDIM(2,NX)=  PONE
      RDIM(3,NX)= -PTWO
      RINIT(NX)=VINIT
C----------------------------------------------------------------------4
      NX=NX+1                                                     !KEQPS
      IF(NX.NE.KEQPS)CALL BOMBED(IAM//' KEQPS pointer wrong')
      NAME(NX)='Equivalent plastic SHEAR strain'
C     EQPS is the time integral of
C           SQRT[2.0]*magnitude[deviator[strain rate]
C     It is defined in this way to be conjugate to ROOTJ2.  In other
C     words, the inner product of the stress deviator with the strain
C     rate is equal to ROOTJ2 times the rate of EQPS.
      KEY(NX)='EQPS'
      IADVCT(NX)=0        ! input and output
C     itype=1  (scalar)
C     RDIM=default        ! units:  dimensionless
      RINIT(NX)=VINIT
C----------------------------------------------------------------------5
      NX=NX+1                                                     !KPW
      IF(NX.NE.KPW)CALL BOMBED(IAM//' KPW pointer wrong')
      NAME(NX)='plastic work'
      KEY(NX)='PW'
      IADVCT(NX)=0        ! output only
      ITYPE(NX)=1
C     itype=1  (scalar)
      RINIT(NX)=VINIT
C-----------------------------------------------------------------------6
      NX=NX+1                                                      !KTMPR
      IF(NX.NE.KT)CALL BOMBED(IAM//' KTMPR pointer wrong')
      NAME(NX)='Temperature'
      KEY(NX)='TMPR'
      IADVCT(NX)=1        ! input/output
      RDIM(4,NX)=  PONE
      IF(SERIAL)THEN
         RINIT(NX)=VINIT
      ELSE
         RINIT(NX)=UI(IPT0)
      ENDIF
C-----------------------------------------------------------------------7
      NX=NX+1                                                     !KSNDSP
      IF(NX.NE.KCS)CALL BOMBED(IAM//'1 KSNDSP pointer wrong')
      NAME(NX)='sound speed'
      KEY(NX)='SNDSP'
      IADVCT(NX)=1        ! input/output
      RDIM(1,NX)=  PONE
      RDIM(3,NX)= -PONE
      IF(SERIAL)THEN
         RINIT(NX)=VINIT
      ELSE
         RINIT(NX)=UI(IPC0)
      ENDIF
C-----------------------------------------------------------------------8
      NX=NX+1                                                       !KRHO
      IF(NX.NE.KRHO)CALL BOMBED(IAM//' KRHO pointer wrong')
      NAME(NX)='Density'
      KEY(NX)='DENS'
      IADVCT(NX)=1        ! input/output
      RDIM(1,NX) = -PTHREE
      RDIM(2,NX) =  PONE
      IF(SERIAL)THEN
         RINIT(NX)=VINIT
      ELSE
         RINIT(NX)=UI(IPRHO0)
      ENDIF
C-----------------------------------------------------------------------9
      NX=NX+1                                                     !KENRGY
      IF(NX.NE.KEU)CALL BOMBED(IAM//' KENRGY pointer wrong')
      NAME(NX)='Energy'
      KEY(NX)='ENRGY'
      IADVCT(NX)=1        ! input/output
      RDIM(1,NX) =  PTWO
      RDIM(2,NX) =  PONE
      RDIM(3,NX) = -PTWO
      IF(SERIAL)THEN
         RINIT(NX)=VINIT
      ELSE
         RINIT(NX)=UI(IPT0)*UI(IPCV)
      ENDIF
C----------------------------------------------------------------------10
      NX=NX+1                                                   !Jacobian
      IF(NX.NE.KRJ)CALL BOMBED(IAM//' KRJ pointer wrong')
      NAME(NX)='Jacobian of deformation'
      KEY(NX)='JACOBIAN'
      IF(SERIAL)THEN
         RINIT(NX)=VINIT
      ELSE
         RINIT(NX)=PONE
      ENDIF
      IADVCT(NX)=1
C----------------------------------------------------------------------11
      NX=NX+1                                                         !AM
      IF(NX.NE.KAM)CALL BOMBED(IAM//' KAM pointer wrong')
      NAME(NX)='Measure of anisotropy'
      KEY(NX)='AM'
      IADVCT(NX)=1
      RINIT(NX)=0.D0
C----------------------------------------------------------------------12
      NX=NX+1                                                       !EQPV
      IF(NX.NE.KEQPV)CALL BOMBED(IAM//' KEQPV pointer wrong')
      NAME(NX)='plastic volume strain'
      KEY(NX)='EQPV'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------13
      NX=NX+1                                                         !F1
      IF(NX.NE.KF1)CALL BOMBED(IAM//' KF1 pointer wrong')
      NAME(NX)='Free EOS ISV 1'
      KEY(NX)='F1'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------14
      NX=NX+1                                                         !F2
      IF(NX.NE.KF2)CALL BOMBED(IAM//' KF2 pointer wrong')
      NAME(NX)='Free EOS ISV 2'
      KEY(NX)='F2'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------15
      NX=NX+1                                                         !F3
      IF(NX.NE.KF3)CALL BOMBED(IAM//' KF3 pointer wrong')
      NAME(NX)='Free EOS ISV 3'
      KEY(NX)='F3'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------16
      NX=NX+1                                                         !F4
      IF(NX.NE.KF4)CALL BOMBED(IAM//' KF4 pointer wrong')
      NAME(NX)='Free EOS ISV 4'
      KEY(NX)='F4'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------17
      NX=NX+1                                                         !F5
      IF(NX.NE.KF5)CALL BOMBED(IAM//' KF5 pointer wrong')
      NAME(NX)='Free EOS ISV 5'
      KEY(NX)='F5'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------18
      NX=NX+1                                                         !F6
      IF(NX.NE.KF6)CALL BOMBED(IAM//' KF6 pointer wrong')
      NAME(NX)='Free EOS ISV 6'
      KEY(NX)='F6'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------19
      NX=NX+1                                                         !F7
      IF(NX.NE.KF7)CALL BOMBED(IAM//' KF7 pointer wrong')
      NAME(NX)='Free EOS ISV 7'
      KEY(NX)='F7'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------20
      NX=NX+1                                                         !F8
      IF(NX.NE.KF8)CALL BOMBED(IAM//' KF8 pointer wrong')
      NAME(NX)='Free EOS ISV 8'
      KEY(NX)='F8'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------21
      NX=NX+1                                                         !F9
      IF(NX.NE.KF9)CALL BOMBED(IAM//' KF9 pointer wrong')
      NAME(NX)='Free EOS ISV 9'
      KEY(NX)='F9'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------22
      NX=NX+1                                                        !F10
      IF(NX.NE.KF10)CALL BOMBED(IAM//' KF10 pointer wrong')
      NAME(NX)='Free EOS ISV 10'
      KEY(NX)='F10'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------23
      NX=NX+1                                                    !KQSSIGXX
      IF(NX.NE.KQSSIGXX)CALL BOMBED(IAM//' KQSSIGXX pointer wrong')
      NAME(NX)='Quasistatic stress'
      KEY(NX)='QSSIGXX'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------24
      NX=NX+1                                                    !KQSSIGYY
      IF(NX.NE.KQSSIGYY)CALL BOMBED(IAM//' KQSSIGYY pointer wrong')
      NAME(NX)='Quasistatic stress'
      KEY(NX)='QSSIGYY'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------25
      NX=NX+1                                                    !KQSSIGZZ
      IF(NX.NE.KQSSIGZZ)CALL BOMBED(IAM//' KQSSIGZZ pointer wrong')
      NAME(NX)='Quasistatic stress'
      KEY(NX)='QSSIGZZ'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------26
      NX=NX+1                                                    !KQSSIGXY
      IF(NX.NE.KQSSIGXY)CALL BOMBED(IAM//' KQSSIGXY pointer wrong')
      NAME(NX)='Quasistatic stress'
      KEY(NX)='EQSSIGXY'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------27
      NX=NX+1                                                   !KQSSIGYZ
      IF(NX.NE.KQSSIGYZ)CALL BOMBED(IAM//' KQSSIGYZ pointer wrong')
      NAME(NX)='Quasistatic stress'
      KEY(NX)='QSSIGYZ'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------28
      NX=NX+1                                                   !KQSSIGZX
      IF(NX.NE.KQSSIGZX)CALL BOMBED(IAM//' KQSSIGZX pointer wrong')
      NAME(NX)='Quasistatic stress'
      KEY(NX)='QSSIGZX'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------29
      NX=NX+1                                                    !KEXX
      IF(NX.NE.KEXX)CALL BOMBED(IAM//' KEXX pointer wrong')
      NAME(NX)='Elastic strain'
      KEY(NX)='EXX'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------30
      NX=NX+1                                                    !KEYY
      IF(NX.NE.KEYY)CALL BOMBED(IAM//' KEYY pointer wrong')
      NAME(NX)='Elastic strain'
      KEY(NX)='EYY'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------31
      NX=NX+1                                                    !KEZZ
      IF(NX.NE.KEZZ)CALL BOMBED(IAM//' KEZZ pointer wrong')
      NAME(NX)='Elastic strain'
      KEY(NX)='EZZ'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------32
      NX=NX+1                                                    !KEXY
      IF(NX.NE.KEXY)CALL BOMBED(IAM//' KEXY pointer wrong')
      NAME(NX)='Elastic strain'
      KEY(NX)='EEXY'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------33
      NX=NX+1                                                   !KEYZ
      IF(NX.NE.KEYZ)CALL BOMBED(IAM//' KEYZ pointer wrong')
      NAME(NX)='Elastic strain'
      KEY(NX)='EYZ'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------34
      NX=NX+1                                                   !KEZX
      IF(NX.NE.KEZX)CALL BOMBED(IAM//' KEZX pointer wrong')
      NAME(NX)='Elastic strain'
      KEY(NX)='EZX'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------35
      NX=NX+1                                                   !KEJ2
      IF(NX.NE.KEJ2)CALL BOMBED(IAM//' KEJ2 pointer wrong')
      NAME(NX)='j2 invariant of elastic strain deviator'
      KEY(NX)='EDEVJ2'
      IADVCT(NX)=1
      RINIT(NX)=VINIT

C     ##################################################################
      IF(NX.GT.NDMMISV)CALL BOMBED
     & ('INCREASE NDMMISV IN ROUTINE DMMRXV AND IN DATA FILE')
C     convert NAME and KEY to character streams NAMEA and KEYA
C     (See note about TOKENS in prolog of this routine)
      CALL TOKENS(NX,NAME,NAMEA)
      CALL TOKENS(NX,KEY ,KEYA )
C     CALL LOGMES('############# exiting DMMRXV')
      RETURN
      END !SUBROUTINE DMMRXV
