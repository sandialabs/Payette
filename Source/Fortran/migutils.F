! Copyright (2011) Sandia Corporation. Under the terms of Contract
! DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains certain
! rights in this software.

! The MIT License

! Copyright (c) Sandia Corporation

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

#ifndef NOPYCALLBACK
C     The following implementations of LOGMES, BOMBED, and FATERR use
C     callback functions to call python functions
C*************************************************************************
      SUBROUTINE LOGMES(MSG)
      CHARACTER*(*) MSG
      EXTERNAL log_message
      CALL log_message(MSG)
      END SUBROUTINE LOGMES
C*************************************************************************
      SUBROUTINE BOMBED(MSG)
      CHARACTER*(*) MSG
      CHARACTER*120 JNKSTR
      EXTERNAL report_and_raise_error
      WRITE(JNKSTR,77)MSG
 77   FORMAT('BOMBED: ',A100)
      CALL report_and_raise_error(JNKSTR)
      END SUBROUTINE BOMBED
C*************************************************************************
      SUBROUTINE FATERR(CALLER, MSG)
      CHARACTER*(*) CALLER
      CHARACTER*(*) MSG
      CHARACTER*300 JNKSTR
      EXTERNAL report_and_raise_error
      WRITE(JNKSTR,77)MSG,CALLER
      CALL report_and_raise_error(JNKSTR)
 77   FORMAT('FATERR: ',A100,' detected by ',A100)
      END SUBROUTINE FATERR
#else
C*************************************************************************
      SUBROUTINE LOGMES(MSG)
      CHARACTER*(*) MSG
      PRINT*,"INFO: "//MSG
      RETURN
      END
C*************************************************************************
      SUBROUTINE BOMBED(MSG)
      CHARACTER*(*) MSG
      PRINT*,"ERROR: "//MSG//" reported from [MIG]"
      STOP
      END
C*************************************************************************
      SUBROUTINE FATERR(CALLER, MSG)
      CHARACTER*(*) CALLER,MSG
      PRINT*,"FATAL ERROR: "//MSG//" reported by ["//CALLER//"]"
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
