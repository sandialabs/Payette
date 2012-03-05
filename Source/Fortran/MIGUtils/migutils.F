#ifndef NOPYCALLBACK
C*************************************************************************
      SUBROUTINE LOGMES(MSG)
      CHARACTER*(*) MSG
      EXTERNAL migMessage
      CALL migMessage(MSG)
      END SUBROUTINE LOGMES
C*************************************************************************
      SUBROUTINE BOMBED(MSG)
      CHARACTER*(*) MSG
      CHARACTER*120 JNKSTR
      EXTERNAL migError
      WRITE(JNKSTR,77)MSG
 77   FORMAT('BOMBED: ',A100)
      CALL migError(JNKSTR)
      END SUBROUTINE BOMBED
C*************************************************************************
      SUBROUTINE FATERR(CALLER, MSG)
      CHARACTER*(*) CALLER
      CHARACTER*(*) MSG
      CHARACTER*300 JNKSTR
      EXTERNAL migError
      WRITE(JNKSTR,77)MSG,CALLER
      CALL migError(JNKSTR)
 77   FORMAT('FATERR: ',A100,' detected by ',A100)
      END SUBROUTINE FATERR
#else
C*************************************************************************
      SUBROUTINE LOGMES(MESS)
      CHARACTER*(*) MESS
      PRINT*,MESS
      RETURN
      END
C*************************************************************************
      SUBROUTINE BOMBED(MSG)
      CHARACTER*(*) MSG
      CHARACTER*200 JNKSTR
      CALL LOGMES('****************************')
      CALL LOGMES('****************************')
      CALL LOGMES('****************************')
      JNKSTR='BOMBED: '// MSG
      CALL LOGMES(JNKSTR)
      STOP
      END
      SUBROUTINE FATERR(CALLER, MSG)
      CHARACTER*200 JNKSTR
      CHARACTER*(*) CALLER,MSG
      CALL LOGMES(' ')
      CALL LOGMES(' ')
      JNKSTR=' //////////////////////////////////////////////////'
      CALL LOGMES(JNKSTR)
      JNKSTR=' ///////////////////  fatal error detected by '//CALLER
      CALL LOGMES(JNKSTR)
      JNKSTR=' //////////////////////////////////////////////////'
      CALL LOGMES(JNKSTR)
      JNKSTR='FATAL ERROR:'//MSG
      CALL LOGMES(JNKSTR)
      CALL LOGMES(' ')
      CALL LOGMES(' ')
      STOP
      STOP
      END
#endif
C***********************************************************************
      SUBROUTINE TOKENS(N,SA,CA)
C    This routine converts the array of strings SA to a single character
C    stream CA with a pipe (|) separating entries.  For example, suppose
C
C              SA(  1) = 'first string             '
C              SA(  2) = 'a witty saying           '
C              SA(  3) = '                         '
C              SA(  4) = 'last                     '
C
C     Then the output of this routine is
C
C             CA = first string|a witty saying||last|
C
C input
C -----
C   N: number of strings in SA (i.e., the dimension of SA)
C   SA: array of strings
C
C output
C ------
C   CA: single character stream of the strings in SA separated by pipes.
C         BEWARE: it is the responsibility of the calling routine to
C         dimension CA at least as large as N*(1+LEN(SA)).
C
C     written: 04/20/95
C     author:  Rebecca Brannon
C
C calling arguments:
      INTEGER N
      CHARACTER*(*) SA(N)
      CHARACTER*(*) CA(N)
C      CHARACTER*1   CA(*)
C local:
      CHARACTER*1 PIPE,BLANK
      PARAMETER (PIPE='|',BLANK=' ')
      INTEGER I,KNT,NCHR,ICHR
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      KNT=0
      DO 502 I=1,N
         DO 500 NCHR=LEN(SA(I)),1,-1
  500    IF(SA(I)(NCHR:NCHR).NE.BLANK) GO TO 7
    7    DO 501 ICHR=1,NCHR
            KNT=KNT+1
            CA(KNT)=SA(I)(ICHR:ICHR)
  501    CONTINUE
         KNT=KNT+1
         CA(KNT)=PIPE
  502 CONTINUE

      RETURN
      END
