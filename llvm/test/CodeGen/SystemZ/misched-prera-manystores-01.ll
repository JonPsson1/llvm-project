; Test FADDs in which the second operand is variable.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

; Check that there are no spills. The many stores should not all end up at
; the bottom.
define void @f1(ptr noalias %src1, ptr noalias %dest) {
; CHECK-LABEL: f1:
; CHECK-NOT: %r15
; CHECK: br %r14
  %val = load <16 x float>, ptr %src1
  %add = fadd <16 x float> %val, %val
  store <16 x float> %add, ptr %dest
  ret void
}
