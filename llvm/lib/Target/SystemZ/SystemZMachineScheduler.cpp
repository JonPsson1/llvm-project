//-- SystemZMachineScheduler.cpp - SystemZ Scheduler Interface -*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// -------------------------- Pre RA scheduling ----------------------------- //
//
//  TODO
//
// -------------------------- Post RA scheduling ---------------------------- //
// SystemZPostRASchedStrategy is a scheduling strategy which is plugged into
// the MachineScheduler. It has a sorted Available set of SUs and a pickNode()
// implementation that looks to optimize decoder grouping and balance the
// usage of processor resources. Scheduler states are saved for the end
// region of each MBB, so that a successor block can learn from it.
//===----------------------------------------------------------------------===//

#include "SystemZMachineScheduler.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineLoopInfo.h"

using namespace llvm;

#define DEBUG_TYPE "machine-scheduler"

/// Pre-RA scheduling ///

static cl::opt<bool> SCHEDCHAINPREDS("sched-chainpreds", cl::Hidden, cl::init(true));

static cl::opt<bool> SCHEDELIMCMP("sched-elimcmp", cl::Hidden, cl::init(false));

static cl::opt<bool> SCHEDPREGCOPYS("sched-pregcopys", cl::Hidden, cl::init(false));
static cl::opt<unsigned> PREGCUSERS("sched-pregc-users", cl::Hidden, cl::init(2));
// Multiple users give some extra COPY elimination, but not that much. Code could
// be simplified somewhat if only the single user case is handled.
// Relative to 1: 2:13% more / 3: 16% more. / 10: 18% more
// ~ or ~
// generic vs 0 (w/out this): 5500 more copies. generic vs 10: 1000 more copies.
// Not sure if any of these values affect performance either way.
// TODO: Could this be a Generic DAGMutation used by other targets also?

// As long as generic sched might be preferred on some regions, this option
// is needed as opposed to handling this in createMachineScheduler().
static cl::opt<bool> GENERICSCHED(
     "generic-sched", cl::Hidden, cl::init(false),
     cl::desc("Run the generic pre-ra scheduler instead of SystemZ "
              "heuristics."));

void SystemZPreRASchedStrategy::
initializePrioRegClasses(const TargetRegisterInfo *TRI_) {
  for (const TargetRegisterClass *RC : TRI_->regclasses()) {
    for (MVT VT : MVT::fp_valuetypes())
      if (TRI_->isTypeLegalForClass(*RC, VT)) {
        PrioRegClasses.insert(RC->getID());
        break;
      }

    // On SystemZ vector and FP registers overlap: add any vector RC.
    if (!PrioRegClasses.count(RC->getID()))
      for (MVT VT : MVT::fp_fixedlen_vector_valuetypes())
        if (TRI_->isTypeLegalForClass(*RC, VT)) {
          PrioRegClasses.insert(RC->getID());
          break;
        }
  }
}

unsigned SystemZPreRASchedStrategy::getRemLat(SchedBoundary *Zone) const {
  if (RemLat == ~0U)
    RemLat = computeRemLatency(*Zone);
  return RemLat;
}

void SystemZPreRASchedStrategy::initializeStoresGroup() {
  StoresGroup.clear();
  FirstStoreInGroupScheduled = false;
  unsigned CurrMaxDepth = 0;
  for (unsigned Idx = DAG->SUnits.size() - 1; Idx + 1 != 0; --Idx) {
    const SUnit *SU = &DAG->SUnits[Idx];
    const MachineInstr *MI = SU->getInstr();
    if (!MI->getNumOperands() || MI->isCopy())
      continue;
    bool HasExplDef = false;
    bool HasVirtUse = false;
    for (unsigned I = 0, E = MI->getNumOperands();
         I != E && I < MI->getDesc().getNumOperands();
         ++I) {
      const MachineOperand &MO = MI->getOperand(I);
      if (!MO.isReg())
        continue;
      if (MO.isDef() && !MO.isDead())
        HasExplDef = true;
      if (MO.isUse() && MO.getReg() && MO.readsReg() &&
          MI->getDesc().operands()[I].OperandType != MCOI::OPERAND_MEMORY)
        HasVirtUse = true;
    }
    bool IsStore = !HasExplDef && HasVirtUse;
    if (SU->getDepth() > CurrMaxDepth) {
      CurrMaxDepth = SU->getDepth();
      bool PrevGroup = StoresGroup.size() > 1;
      StoresGroup.clear();
      if (PrevGroup)
        return; // Skip regions with multiple groups of different depths.
      if (IsStore)
        StoresGroup.insert(SU);
    }
    else if (IsStore && !StoresGroup.empty() && SU->getDepth() == CurrMaxDepth) {
      // The group members should all have the same opcode.
      if ((*StoresGroup.begin())->getInstr()->getOpcode() != MI->getOpcode()) {
        StoresGroup.clear();
        return;
      }
      StoresGroup.insert(SU);
    }
  }

  // Value of 8 handles a known regression (with group of 20). TODO: Is 4 or
  // 6 slightly better, perhaps?
  if (StoresGroup.size() < 8)
    StoresGroup.clear();
}

void SystemZPreRASchedStrategy::initPolicy(MachineBasicBlock::iterator Begin,
                                           MachineBasicBlock::iterator End,
                                           unsigned NumRegionInstrs) {
  // Use GenericScheduler if requested on CL or for Z10, which has no sched
  // model.
  auto &ST = Begin->getMF()->getSubtarget();
  DoGenericSched = GENERICSCHED ||  // .getNumOccurrences() ?
    !ST.getSchedModel().hasInstrSchedModel();
  if (DoGenericSched) {
    GenericScheduler::initPolicy(Begin, End, NumRegionInstrs);
    return;
  }

  // Keep track of live regs instead of using the generic reg pressure tracking.
  RegionPolicy.ShouldTrackPressure = false;
  // These heuristics has so far seemed to work better without adding a
  // top-down boundary.
  RegionPolicy.OnlyBottomUp = true;
}

void SystemZPreRASchedStrategy::initialize(ScheduleDAGMI *dag) {
  Bot.PreRAHazardChecks = DoGenericSched;

  GenericScheduler::initialize(dag);
  if (DoGenericSched)
    return;

  NumLeft = DAG->SUnits.size();
  RemLat = ~0U;

  // It seems to work well to include the latencies in this heuristic (as
  // opposed to something like a "unit DAG height" with all latencies counted
  // as 1).
  unsigned DAGHeight = 0;
  for (unsigned Idx = 0, End = DAG->SUnits.size(); Idx != End; ++Idx)
    DAGHeight = std::max(DAGHeight, DAG->SUnits[Idx].getHeight());
  IsWideDAG = DAG->SUnits.size() >= 3 * std::max(DAGHeight, 1u);

  LiveRegs.clear();
  for (unsigned I = 0, E = DAG->MRI.getNumVirtRegs(); I != E; ++I) {
    Register VirtReg = Register::index2VirtReg(I);
    const LiveInterval &LI = DAG->getLIS()->getInterval(VirtReg);
    LiveQueryResult LRQ =
      LI.Query(DAG->getLIS()->
               getInstructionIndex(*DAG->SUnits.back().getInstr()));
    if (LRQ.valueOut())
      LiveRegs.insert(VirtReg);
  }
  LLVM_DEBUG( dbgs() << " Live out at bottom: ";
              for (auto Reg : LiveRegs)
                dbgs() << "%" << Reg.virtRegIndex() << ", ";
              dbgs() << "\n";);

  IsRedefining = std::vector<bool>(DAG->SUnits.size(), false);
  for (unsigned Idx = 0, End = DAG->SUnits.size(); Idx != End; ++Idx) {
    const MachineInstr *MI = DAG->SUnits[Idx].getInstr();
    if (MI->getNumOperands()) {
      const MachineOperand &DefMO = MI->getOperand(0);
      if (DefMO.isReg() && DefMO.isDef() && DefMO.getReg().isVirtual())
        IsRedefining[Idx] = MI->readsVirtualRegister(DefMO.getReg());
    }
  }

  // Only chain preds. To stack: Mostly store immediate, but also some
  // memmoves.  These do not use registers, but if scheduled they can make
  // more instructions available due to memory deps.
  if (SCHEDCHAINPREDS) {
    HasOnlyChainPreds = std::vector<bool>(DAG->SUnits.size(), false);
    for (unsigned Idx = 0, End = DAG->SUnits.size(); Idx != End; ++Idx) {
      const SUnit *SU = &DAG->SUnits[Idx];
      const MachineInstr *MI = SU->getInstr();
      bool HasRegOp = false;
      if (SU->getInstr()->mayStore()) {
        for (auto &MO : MI->operands())
          if (MO.isReg() && MO.getReg() && MO.readsReg() &&
              (MO.getReg().isVirtual() || DAG->MRI.isAllocatable(MO.getReg()))) {
            HasRegOp = true;
            break;
          }
        bool HasChainPred = false;
        for (const SDep &Pred : SU->Preds)
          if (Pred.getKind() == SDep::Order && !Pred.getSUnit()->isBoundaryNode()) {
            HasChainPred = true;
            break;
          }
        HasOnlyChainPreds[Idx] = !HasRegOp && HasChainPred;
      }
    }
  }

  initializeStoresGroup();
  if (SCHEDELIMCMP)
    initializeCmpElim();
  if (SCHEDPREGCOPYS)
    initializePRegDeps();
}

// CmpElim
static bool touchesCC(const MachineInstr *MI) {
  return MI->getDesc().hasImplicitUseOfPhysReg(SystemZ::CC) ||
    MI->getDesc().hasImplicitDefOfPhysReg(SystemZ::CC);
}

void SystemZPreRASchedStrategy::schedNode(SUnit *SU, bool IsTopNode) {
  unsigned ExpectedLatencyIn = Bot.getExpectedLatency();
  GenericScheduler::schedNode(SU, IsTopNode);
  if (DoGenericSched)
    return;

  LLVM_DEBUG( dbgs() << "Live regs was: ";
              for (auto R : LiveRegs)
                dbgs() << "%" << R.virtRegIndex() << ", ";
              dbgs() << "\n";);

  if (!FirstStoreInGroupScheduled && StoresGroup.count(SU))
    FirstStoreInGroupScheduled = true;

  MachineInstr *MI = SU->getInstr();
  for (auto &MO : MI->operands())
    if (MO.isReg() && MO.getReg() && !MO.isImplicit() &&
        Register::isVirtualRegister(MO.getReg())) {
      if (MO.isDef()) {  // XXX there are implicit uses of vreg:subreg, so the subreg def may not be live:
        assert(LiveRegs.count(MO.getReg()) || MO.isDead() || MO.getSubReg());
        if (!IsRedefining[SU->NodeNum])
          LiveRegs.erase(MO.getReg());
      }
      else if (MO.readsReg())
        LiveRegs.insert(MO.getReg());
    }
  assert(NumLeft > 0);
  --NumLeft;
  RemLat = ~0U;

  if (SCHEDELIMCMP) {
    if (Cmp2Src.count(SU)) // A Cmp was just scheduled.
      checkCmpSrcForCmpElim(SU);
    else if (SU == CmpSrcSU) { // CmpSrc was just scheduled.
      CmpSrcSU = nullptr;
      // Moving the CmpSrcSU around normal heuristics should affect other nodes
      // as little as possible.
      if (CmpSrcPref) {
        Bot.resetExpectedLatency(ExpectedLatencyIn);
        CmpSrcPref = 0;
      }
    }
    else if (touchesCC(MI)) { // Any other instruction that uses/defs CC.
      CmpSrcSU = nullptr;
      CmpSrcPref = 0;
    }
    else if (CmpSrcPref > 0)
      --CmpSrcPref;
  }
}

static int biasPhysRegExtra(const SUnit *SU, bool isTop) {
  if (int Res = biasPhysReg(SU, isTop))
    return Res;

  // Load Address (stack) (load immediate recognized in biasPhysReg).
  const MachineInstr *MI = SU->getInstr();
  if (MI->getNumOperands() && !MI->isCopy()) {
    const MachineOperand &DefMO = MI->getOperand(0);
    if (DefMO.isReg() && DefMO.isDef() && DefMO.getReg().isPhysical()) {
      bool DoBias = true;
      for (unsigned I = 1, E = MI->getNumOperands(); I != E; ++I) {
        const MachineOperand &Op = MI->getOperand(I);
        if (Op.isReg() && Op.getReg()) {
          DoBias = false;
          break;
        }
      }
      if (DoBias)
        return isTop ? -1 : 1;
    }
  }

  return 0;
}

int SystemZPreRASchedStrategy::
computeSULivenessScore(SchedCandidate &C, ScheduleDAGMILive *DAG,
                       SchedBoundary *Zone) const {
  // Not all data deps are modelled around the SUnit - some data edges near
  // boundaries are missing: Look directly at the MI operands instead.
  const SUnit *SU = C.SU;
  const MachineInstr *MI = SU->getInstr();
  if (!MI->getNumOperands() || MI->isCopy())
    return 0;

  unsigned PrioKills = 0;
  unsigned GPRKills = 0;
  unsigned AddrKills = 0;
  bool HasPrioUse = false;
  for (unsigned I = 0, E = MI->getNumOperands();
       I != E && I < MI->getDesc().getNumOperands();
       ++I) {
    const MachineOperand &MO = MI->getOperand(I);
    if (!MO.isReg() || !MO.isUse() || !MO.readsReg() ||
        !MO.getReg().isVirtual())
      continue;
    HasPrioUse |= isPrioVirtReg(MO.getReg(), &DAG->MRI);
    if (LiveRegs.count(MO.getReg()))
      continue;
    if (isPrioVirtReg(MO.getReg(), &DAG->MRI))
      PrioKills++;
    else if (MI->getDesc().operands()[I].OperandType != MCOI::OPERAND_MEMORY)
      GPRKills++;
    else
      AddrKills++;
  }

  const MachineOperand &DefMO = MI->getOperand(0);
  assert(!(DefMO.isReg() && DefMO.isDef() && DefMO.getReg().isPhysical()) &&
         "Did not expect physreg def!");

  // Schedule SU next if all (prioritized) uses are already live and the
  // scheduled latency is not increased.
  bool IsLoad = DefMO.isReg() && DefMO.isDef() && !DefMO.isDead() &&
    !IsRedefining[SU->NodeNum];
  bool PreservesSchedLat = SU->getHeight() <= Zone->getScheduledLatency();
  // Care more about FP: Ignore GPR/Addr kills with an FP def.
  bool UsesLivePrio = IsLoad && !PrioKills &&
    (isPrioVirtReg(DefMO.getReg(), &DAG->MRI) || GPRKills + AddrKills == 0);
  const unsigned Cycles = 2;
  unsigned Margin = SchedModel->getIssueWidth() * (Cycles + SU->Latency - 1);
  bool HasDistToTop = NumLeft > Margin;
  if (PreservesSchedLat && UsesLivePrio)
    ;
  // If there will be relatively many SUs scheduled above this one it should
  // not be a problem to increase the scheduled latency given the OOO
  // execution. Seems best to demand all regs to be live here.
  else if (HasDistToTop && !PrioKills && !GPRKills && !AddrKills)
    ;
  else
    IsLoad = false;

  // This handles regions with many chained stores of the same depth at the
  // bottom in the input order (cactus).
  // TODO:
  // - Experiment further: use RemLatency/Depth with single stores?
  // - Currently avoiding DefMO (it may be that there are two non-live uses).
  // - Schedule stores of live-though regs early (low)?
  // - check if there are (many) chain-only-pred(s) waiting on just SU?
  //   Maybe scheduling SU would allow scheduling predecessor load(s).
  bool IsStore = (!DefMO.isReg() || !DefMO.isDef() || DefMO.isDead());
  IsStore &= (PrioKills > 0 || (!HasPrioUse && GPRKills > 0));
  IsStore &= FirstStoreInGroupScheduled && StoresGroup.count(SU);

  // TODO: Would it give further benefits to schedule small subtrees as a
  // unit when this would reduce register pressure?
  if (IsLoad)
    return -1;

  if (IsStore)
    return 1;

  return 0;
}

bool SystemZPreRASchedStrategy::tryCandidate(SchedCandidate &Cand,
                                             SchedCandidate &TryCand,
                                             SchedBoundary *Zone) const {
  if (DoGenericSched)
    return GenericScheduler::tryCandidate(Cand, TryCand, Zone);

  assert(Zone && !Zone->isTop() && "Bottom-Up scheduling only.");

  // Initialize the candidate if needed. (From GenericScheduler)
  if (!Cand.isValid()) {
    TryCand.Reason = FirstValid;
    return true;
  }

  // Bias PhysReg Defs and copies to their uses and defined respectively.
  if (tryGreater(biasPhysRegExtra(TryCand.SU, TryCand.AtTop),
                 biasPhysRegExtra(Cand.SU, Cand.AtTop), TryCand, Cand, PhysReg))
    return TryCand.Reason != NoCand;

  bool SkipPhysRegs = biasPhysRegExtra(TryCand.SU, TryCand.AtTop) &&
    biasPhysRegExtra(Cand.SU, TryCand.AtTop);
  if (SkipPhysRegs) { // Both biased same way.  XXX worthwhile?  Whatif one long-latency...
    tryGreater(TryCand.SU->NodeNum, Cand.SU->NodeNum, TryCand, Cand,
               NodeOrder);
    return TryCand.Reason != NoCand;
  }

  if (SCHEDCHAINPREDS)
    if (tryGreater(HasOnlyChainPreds[TryCand.SU->NodeNum],
                   HasOnlyChainPreds[Cand.SU->NodeNum],
                   TryCand, Cand, ChainReduce))
      return TryCand.Reason != NoCand;

  int TryCandScore = computeSULivenessScore(TryCand, DAG, Zone);
  int CandScore = computeSULivenessScore(Cand, DAG, Zone);
  if (tryLess(TryCandScore, CandScore, TryCand, Cand, LivenessReduce))
    return TryCand.Reason != NoCand;

  if (SCHEDELIMCMP && tryCmpElimOrdering(TryCand, Cand))
    return TryCand.Reason != NoCand;

  if (SCHEDPREGCOPYS) {
    // If a candidate is a user of a vreg that is defined by a copy from a
    // preg, schedule it above some other node that is defining a vreg that
    // will be copied to the same preg.
    bool TryPRegDepSucc = hasPRegDepSuccessor(TryCand.SU);
    bool PRegDepSucc = hasPRegDepSuccessor(Cand.SU);
    if (tryLess(TryPRegDepSucc, PRegDepSucc, TryCand, Cand, Weak))
      return TryCand.Reason != NoCand;
  }

  if (!IsWideDAG && TryCand.SU->getHeight() != Cand.SU->getHeight() &&
      (std::max(TryCand.SU->getHeight(), Cand.SU->getHeight()) >
       Zone->getScheduledLatency())) {
    unsigned HigherSUDepth = TryCand.SU->getHeight() < Cand.SU->getHeight() ?
      Cand.SU->getDepth() : TryCand.SU->getDepth();
    if (HigherSUDepth != getRemLat(Zone) &&
        tryLess(TryCand.SU->getHeight(), Cand.SU->getHeight(),
                TryCand, Cand, GenericSchedulerBase::BotHeightReduce)) {
      return TryCand.Reason != NoCand;
    }
  }

  // Weak edges are for clustering and other constraints.
  if (tryLess(TryCand.SU->WeakSuccsLeft, Cand.SU->WeakSuccsLeft,
              TryCand, Cand, Weak))
    return TryCand.Reason != NoCand;

  // Fall through to original instruction order.
  if (TryCand.SU->NodeNum > Cand.SU->NodeNum) {
    TryCand.Reason = NodeOrder;
    return true;
  }

  return false;
}

//////////////////////  CmpElim  /////////////////////

static bool hasUsableCCDef(const SUnit *CmpSrcSU, const SUnit *CmpSU,
                           SmallVector<MachineInstr *, 4> &CCUsers,
                           const SystemZInstrInfo *TII) {
  MachineInstr *CmpSrcMI = CmpSrcSU->getInstr();
  MachineInstr *CmpMI = CmpSU->getInstr();

  if (unsigned ConvOpc = TII->getLoadAndTest(CmpSrcMI->getOpcode()))
    if (TII->adjustCCMasksForInstr(*CmpSrcMI, *CmpMI, CCUsers,
                                   ConvOpc, /*DoAdjust=*/false))
      return true;
  if (TII->adjustCCMasksForInstr(*CmpSrcMI, *CmpMI, CCUsers,
                                 /*ConvOpc=*/0, /*DoAdjust=*/false))
    return true;
  if (unsigned ConvOpc = TII->getConvertToLogicalOpcode(CmpSrcMI->getOpcode()))
    if (TII->adjustCCMasksForInstr(*CmpSrcMI, *CmpMI, CCUsers,
                                   ConvOpc, /*DoAdjust=*/false))
      return true;
  return false;
}

static bool isCCLiveAfter(const MachineInstr *MI,
                          const MachineBasicBlock *MBB) {
  for (MachineBasicBlock::const_iterator II = std::next(MI->getIterator());
       II != MBB->end(); ++II) {
    if (II->getDesc().hasImplicitUseOfPhysReg(SystemZ::CC))
      return true;
    if (II->getDesc().hasImplicitDefOfPhysReg(SystemZ::CC))
      return false;
  }

  for (const MachineBasicBlock *Succ : MBB->successors())
    if (Succ->isLiveIn(SystemZ::CC))
      return true;

  return false;
}

void SystemZPreRASchedStrategy::initializeCmpElim() {
  const SystemZInstrInfo *TII = static_cast<const SystemZInstrInfo *>(DAG->TII);
  unsigned DAGSize = DAG->SUnits.size();

  // Special handling of bottom of region which could have an ExitSU.
  MachineInstr *LastMI = DAG->SUnits[DAGSize - 1].getInstr();
  const MachineBasicBlock *MBB = LastMI->getParent();
  if (MachineInstr *ExitMI = DAG->ExitSU.getInstr())
    LastMI = ExitMI;
  else {
    assert(LastMI == &MBB->instr_back());
    assert(!LastMI->isTerminator() && !LastMI->isBranch() && !LastMI->isCall());
  }

  SmallVector<MachineInstr *, 4> CCUsers;
  bool CompleteCCUsers = true;
  if (!LastMI->getDesc().hasImplicitDefOfPhysReg(SystemZ::CC) &&
      isCCLiveAfter(LastMI, MBB))
    CompleteCCUsers = false;
  else if (LastMI->getDesc().hasImplicitUseOfPhysReg(SystemZ::CC))
    CCUsers.push_back(LastMI);

  unsigned Idx = DAGSize - 1;
  if (!CompleteCCUsers)
    while (Idx + 1 != 0) {
      SUnit *SU = &DAG->SUnits[Idx--];
      if (SU->getInstr()->getDesc().hasImplicitDefOfPhysReg(SystemZ::CC))
        break;
    }

  // Add mappings from compares to their definitions of source values in
  // cases where comparison elimination is possible.
  Cmp2Src.clear();
  for (;Idx + 1 != 0; --Idx) {
    SUnit *SU = &DAG->SUnits[Idx];
    MachineInstr *MI = SU->getInstr();
    if (TII->isCompareZero(*MI)) {
      assert(!CCUsers.empty() && "Cmp should have at least one CC user.");
      SUnit *SrcSU = nullptr; // Src could be live-in.
      for (const SDep &Pred : SU->Preds)
        if (Pred.getKind() == SDep::Data) {
          assert(Pred.getReg() == TII->getCompareSourceReg(*MI) &&
                 "CmpSU data-edge should reflect its source register.");
          SrcSU = Pred.getSUnit();
        }
      if (SrcSU && hasUsableCCDef(SrcSU, SU, CCUsers, TII))
        Cmp2Src[SU] = SrcSU;
      CCUsers.clear();
      continue;
    }
    if (MI->getDesc().hasImplicitDefOfPhysReg(SystemZ::CC))
      CCUsers.clear();
    if (MI->getDesc().hasImplicitUseOfPhysReg(SystemZ::CC))
      CCUsers.push_back(MI);
  }

  CmpSrcSU = nullptr;
  CmpSrcPref = 0;
}

// Returns false if a reached successor clobbers CC.
static bool findSuccs(const SUnit *RootSU,
                      const SUnit *CmpSU,
                      const SUnit *SU,
                      std::set<const SUnit *> &Succs) {
  if (SU != RootSU) {
    if (SU->isScheduled)
      return true;
    if (SU->getInstr()->getDesc().hasImplicitDefOfPhysReg(SystemZ::CC))
      return false;
    if (!Succs.insert(SU).second)
      return true;
  }
  for (const SDep &Succ : SU->Succs) {
    const SUnit *SuccSU = Succ.getSUnit();
    if (!SuccSU->isBoundaryNode() && SuccSU != CmpSU &&
        !findSuccs(RootSU, CmpSU, SuccSU, Succs))
      return false;
  }
  return true;
}

static void findPreds(const SUnit *RootSU,
                      const SUnit *SU,
                      std::set<const SUnit *> &Preds) {
  if (SU->isScheduled /*weak dependency*/)
    return;
  if (SU != RootSU && !Preds.insert(SU).second)
    return;
  for (const SDep &Pred : SU->Preds)
    if (!Pred.getSUnit()->isBoundaryNode())
      findPreds(RootSU, Pred.getSUnit(), Preds);
}

bool SystemZPreRASchedStrategy::
tryCmpElimOrdering(SchedCandidate &TryCand, SchedCandidate &Cand) const {
  bool TryCandIsCmpSrc = TryCand.SU == CmpSrcSU;
  bool CandIsCmpSrc = Cand.SU == CmpSrcSU;

  if (CmpSrcSU) {
    // CmpSrc <> CC: CmpSrc needs to go below if other def/use CC.
    if (tryGreater(TryCandIsCmpSrc && touchesCC(Cand.SU->getInstr()),
                   CandIsCmpSrc && touchesCC(TryCand.SU->getInstr()),
                   TryCand, Cand, GenericSchedulerBase::CmpCC))
      return true;

    // CmpSrc <> No-CC: CmpSrc has some latency so move it up.
    if (CmpSrcPref &&
        tryLess(TryCandIsCmpSrc, CandIsCmpSrc, TryCand, Cand,
                GenericSchedulerBase::CmpCC))
      return true;

    // CC <> No-CC: Try to move a CC def/use up.
    if (tryLess(touchesCC(TryCand.SU->getInstr()),
                touchesCC(Cand.SU->getInstr()), TryCand, Cand,
                GenericSchedulerBase::CmpCC))
      return true;
  }

  return false;
}

bool SystemZPreRASchedStrategy::checkCmpSrcForCmpElim(SUnit *CmpSU) {
  CmpSrcSU = nullptr;
  CmpSrcPref = 0;

  SUnit *SrcSU = Cmp2Src[CmpSU];
  // In-order heuristic for top of (or small) region. With bigger regions
  // this probably isn't very important at the bottom.
  unsigned IssueW = DAG->getSchedModel()->getIssueWidth();
  const unsigned TopRegionCycles = 3;  // XXX 5?
  if (NumLeft <= TopRegionCycles * IssueW) {

    std::set<const SUnit *> Succs;
    if (!findSuccs(SrcSU, CmpSU, SrcSU, Succs))
      return false;

    std::set<const SUnit *> Preds;
    findPreds(SrcSU, SrcSU, Preds);
    unsigned NumPreExistingPreds = Preds.size();
    // Any instruction above CmpSU clobbering CC would have to become
    // predecessors of CmpSrc, which might make this not worthwhile.
    for (unsigned Idx = CmpSU->NodeNum - 1; Idx + 1 != 0; --Idx) {
      SUnit *SU = &DAG->SUnits[Idx];
      if (SU->isScheduled || SU == SrcSU || Succs.count(SU) ||
          Preds.count(SU) || SU == CmpSU)
        continue;
      if (touchesCC(SU->getInstr()))
        findPreds(SrcSU, SU, Preds);
    }
    unsigned BestCycle0 = NumPreExistingPreds / IssueW;
    unsigned BestCycle1 = Preds.size() / IssueW;
    assert(BestCycle1 >= BestCycle0);
    unsigned CyclesLost = std::min(BestCycle1 - BestCycle0,
                                   uint32_t(SrcSU->Latency));

    // The latency of the compare to be elimated is factored in, but only if
    // there are no other sucessors of SrcSU of the same height as CmpSU.
    unsigned CyclesSaved = 0;
    unsigned NoCmpSuccHeight = 0;
    for (const SDep &Succ : SrcSU->Succs)
      if (Succ.getSUnit() != CmpSU)
        NoCmpSuccHeight = std::max(NoCmpSuccHeight, Succ.getSUnit()->getHeight());
    if (NoCmpSuccHeight < CmpSU->getHeight())
      CyclesSaved = std::min(uint32_t(CmpSU->Latency),
                             CmpSU->getHeight() - NoCmpSuccHeight);

    // Adjust schedule for the compare only if no any additional stall
    // results.  Ignoring for now the latency difference of a converted
    // CmpSrc (e.g. LG -> LTG).
    if (CyclesLost <= CyclesSaved) {
      CmpSrcSU = SrcSU;
      // The compare was just scheduled. Help the CmpSrc up in schedule with
      // its latency minus the latency of the Cmp which will eventually be
      // eliminated. XXX Worthwhile?
      if (SrcSU->Latency > CmpSU->Latency)
        CmpSrcPref = std::min((SrcSU->Latency - CmpSU->Latency) * IssueW,
                              NumLeft);
      return true;
    }
  }

  return false;
}

/////////////////////  pregcopys  ////////////////////

static bool isCopyWithPreg(const SUnit *SU, unsigned OpIdx) {
  const MachineInstr *MI = SU->getInstr();
  return MI->isCopy() && MI->getOperand(OpIdx).getReg().isPhysical();
}

void SystemZPreRASchedStrategy::initializePRegDeps() {
  PRegUse2DefDep.clear();
  PRegUser2UsersGroup.clear();
  std::map<Register, std::vector<const SUnit *> > SrcPReg2Users;
  std::map<Register, const SUnit *> DstPReg2Def;
  // Try to schedule vreg0 above vreg1 to allow both be allocated to preg0.
  //   vreg0 = COPY preg0
  //   ...   = vreg0:use    <= SrsPReg2Users[preg0]
  //   vreg1 = ...          <= DstPReg2Def[preg0]
  //   preg0 = COPY vreg1
  for (unsigned Idx = 0, End = DAG->SUnits.size(); Idx != End; ++Idx) {
    const SUnit *SU = &DAG->SUnits[Idx];
    const MachineInstr *MI = SU->getInstr();
    assert(!MI->isCall() && "Expected to treat calls as scheduling boundaries.");
    if (!MI->isCopy())
      continue;

    Register DstReg = MI->getOperand(0).getReg();
    Register SrcReg = MI->getOperand(1).getReg();
    if (SrcReg.isPhysical() && DAG->MRI.isAllocatable(SrcReg) &&
        DstReg.isVirtual()) {
      if (LiveRegs.count(DstReg)) // Live out of region.
        continue;
      for (const SDep &Succ : SU->Succs)
        if (Succ.getKind() == SDep::Data) {
          if (isCopyWithPreg(Succ.getSUnit(), 0)) {
            // Don't bother if it is copied to a preg (in bottom of region).
            SrcPReg2Users[SrcReg].clear();
            break;
          }
          SrcPReg2Users[SrcReg].push_back(Succ.getSUnit());
        }
    }
    else if (DstReg.isPhysical() && DAG->MRI.isAllocatable(DstReg) &&
             SrcReg.isVirtual()) {
      const SUnit *VRegDefSU = nullptr;
      for (const SDep &Pred : SU->Preds)
        if (Pred.getKind() == SDep::Data) {
          VRegDefSU = Pred.getSUnit();
          break;
        }
      if (!VRegDefSU || // Live into region.
          isCopyWithPreg(VRegDefSU, 1)) // copy from a preg (in top of region).
        continue;
      DstPReg2Def[DstReg] = VRegDefSU;
    }
  }

  for (auto II : SrcPReg2Users) {
    Register PReg = II.first;
    std::vector<const SUnit *> &PRegUsers = II.second;
    if (PRegUsers.size() > PREGCUSERS)
      continue;

    if (DstPReg2Def.find(PReg) != DstPReg2Def.end()) {
      const SUnit *VRegDef = DstPReg2Def[PReg];
      for (auto *PRegUser : PRegUsers) {
        if (VRegDef->NodeNum == PRegUser->NodeNum)
          continue;
        // Try to put VRegDef below PRegUser during scheduling.
        PRegUse2DefDep[PRegUser] = VRegDef;
        // multiple users
        if (PRegUsers.size() > 1)
          for (auto *User : PRegUsers)
            PRegUser2UsersGroup[PRegUser].push_back(User);
      }
    }
  }
}

bool SystemZPreRASchedStrategy::hasPRegDepSuccessor(const SUnit *SU) const {
  auto II = PRegUse2DefDep.find(SU);
  if (II == PRegUse2DefDep.end())
    return 0;

  // The other node has already been scheduled below.
  const SUnit *PRegDefSU = II->second;
  if (PRegDefSU->isScheduled)
    return 0;

  // multiple users
  // If any other user has been scheduled, it doesn't matter.
  auto Itr = PRegUser2UsersGroup.find(SU);
  if (Itr != PRegUser2UsersGroup.end()) {
    const std::vector<const SUnit*> &UserGroup = Itr->second;
    for (auto *User : UserGroup)
      if (User->isScheduled)
        return 0;
  }

  return 1;
}


/// Post-RA scheduling ///

#ifndef NDEBUG
// Print the set of SUs
void SystemZPostRASchedStrategy::SUSet::
dump(SystemZHazardRecognizer &HazardRec) const {
  dbgs() << "{";
  for (auto &SU : *this) {
    HazardRec.dumpSU(SU, dbgs());
    if (SU != *rbegin())
      dbgs() << ",  ";
  }
  dbgs() << "}\n";
}
#endif

// Try to find a single predecessor that would be interesting for the
// scheduler in the top-most region of MBB.
static MachineBasicBlock *getSingleSchedPred(MachineBasicBlock *MBB,
                                             const MachineLoop *Loop) {
  MachineBasicBlock *PredMBB = nullptr;
  if (MBB->pred_size() == 1)
    PredMBB = *MBB->pred_begin();

  // The loop header has two predecessors, return the latch, but not for a
  // single block loop.
  if (MBB->pred_size() == 2 && Loop != nullptr && Loop->getHeader() == MBB) {
    for (MachineBasicBlock *Pred : MBB->predecessors())
      if (Loop->contains(Pred))
        PredMBB = (Pred == MBB ? nullptr : Pred);
  }

  assert ((PredMBB == nullptr || !Loop || Loop->contains(PredMBB))
          && "Loop MBB should not consider predecessor outside of loop.");

  return PredMBB;
}

void SystemZPostRASchedStrategy::
advanceTo(MachineBasicBlock::iterator NextBegin) {
  MachineBasicBlock::iterator LastEmittedMI = HazardRec->getLastEmittedMI();
  MachineBasicBlock::iterator I =
    ((LastEmittedMI != nullptr && LastEmittedMI->getParent() == MBB) ?
     std::next(LastEmittedMI) : MBB->begin());

  for (; I != NextBegin; ++I) {
    if (I->isPosition() || I->isDebugInstr())
      continue;
    HazardRec->emitInstruction(&*I);
  }
}

void SystemZPostRASchedStrategy::initialize(ScheduleDAGMI *dag) {
  Available.clear();  // -misched-cutoff.
  LLVM_DEBUG(HazardRec->dumpState(););
}

void SystemZPostRASchedStrategy::enterMBB(MachineBasicBlock *NextMBB) {
  assert ((SchedStates.find(NextMBB) == SchedStates.end()) &&
          "Entering MBB twice?");
  LLVM_DEBUG(dbgs() << "** Entering " << printMBBReference(*NextMBB));

  MBB = NextMBB;

  /// Create a HazardRec for MBB, save it in SchedStates and set HazardRec to
  /// point to it.
  HazardRec = SchedStates[MBB] = new SystemZHazardRecognizer(TII, &SchedModel);
  LLVM_DEBUG(const MachineLoop *Loop = MLI->getLoopFor(MBB);
             if (Loop && Loop->getHeader() == MBB) dbgs() << " (Loop header)";
             dbgs() << ":\n";);

  // Try to take over the state from a single predecessor, if it has been
  // scheduled. If this is not possible, we are done.
  MachineBasicBlock *SinglePredMBB =
    getSingleSchedPred(MBB, MLI->getLoopFor(MBB));
  if (SinglePredMBB == nullptr)
    return;
  auto It = SchedStates.find(SinglePredMBB);
  if (It == SchedStates.end())
    return;

  LLVM_DEBUG(dbgs() << "** Continued scheduling from "
                    << printMBBReference(*SinglePredMBB) << "\n";);

  HazardRec->copyState(It->second);
  LLVM_DEBUG(HazardRec->dumpState(););

  // Emit incoming terminator(s). Be optimistic and assume that branch
  // prediction will generally do "the right thing".
  for (MachineInstr &MI : SinglePredMBB->terminators()) {
    LLVM_DEBUG(dbgs() << "** Emitting incoming branch: "; MI.dump(););
    bool TakenBranch = (MI.isBranch() &&
                        (TII->getBranchInfo(MI).isIndirect() ||
                         TII->getBranchInfo(MI).getMBBTarget() == MBB));
    HazardRec->emitInstruction(&MI, TakenBranch);
    if (TakenBranch)
      break;
  }
}

void SystemZPostRASchedStrategy::leaveMBB() {
  LLVM_DEBUG(dbgs() << "** Leaving " << printMBBReference(*MBB) << "\n";);

  // Advance to first terminator. The successor block will handle terminators
  // dependent on CFG layout (T/NT branch etc).
  advanceTo(MBB->getFirstTerminator());
}

SystemZPostRASchedStrategy::
SystemZPostRASchedStrategy(const MachineSchedContext *C)
  : MLI(C->MLI),
    TII(static_cast<const SystemZInstrInfo *>
        (C->MF->getSubtarget().getInstrInfo())),
    MBB(nullptr), HazardRec(nullptr) {
  const TargetSubtargetInfo *ST = &C->MF->getSubtarget();
  SchedModel.init(ST);
}

SystemZPostRASchedStrategy::~SystemZPostRASchedStrategy() {
  // Delete hazard recognizers kept around for each MBB.
  for (auto I : SchedStates) {
    SystemZHazardRecognizer *hazrec = I.second;
    delete hazrec;
  }
}

void SystemZPostRASchedStrategy::initPolicy(MachineBasicBlock::iterator Begin,
                                            MachineBasicBlock::iterator End,
                                            unsigned NumRegionInstrs) {
  // Don't emit the terminators.
  if (Begin->isTerminator())
    return;

  // Emit any instructions before start of region.
  advanceTo(Begin);
}

// Pick the next node to schedule.
SUnit *SystemZPostRASchedStrategy::pickNode(bool &IsTopNode) {
  // Only scheduling top-down.
  IsTopNode = true;

  if (Available.empty())
    return nullptr;

  // If only one choice, return it.
  if (Available.size() == 1) {
    LLVM_DEBUG(dbgs() << "** Only one: ";
               HazardRec->dumpSU(*Available.begin(), dbgs()); dbgs() << "\n";);
    return *Available.begin();
  }

  // All nodes that are possible to schedule are stored in the Available set.
  LLVM_DEBUG(dbgs() << "** Available: "; Available.dump(*HazardRec););

  Candidate Best;
  for (auto *SU : Available) {

    // SU is the next candidate to be compared against current Best.
    Candidate c(SU, *HazardRec);

    // Remeber which SU is the best candidate.
    if (Best.SU == nullptr || c < Best) {
      Best = c;
      LLVM_DEBUG(dbgs() << "** Best so far: ";);
    } else
      LLVM_DEBUG(dbgs() << "** Tried      : ";);
    LLVM_DEBUG(HazardRec->dumpSU(c.SU, dbgs()); c.dumpCosts();
               dbgs() << " Height:" << c.SU->getHeight(); dbgs() << "\n";);

    // Once we know we have seen all SUs that affect grouping or use unbuffered
    // resources, we can stop iterating if Best looks good.
    if (!SU->isScheduleHigh && Best.noCost())
      break;
  }

  assert (Best.SU != nullptr);
  return Best.SU;
}

SystemZPostRASchedStrategy::Candidate::
Candidate(SUnit *SU_, SystemZHazardRecognizer &HazardRec) : Candidate() {
  SU = SU_;

  // Check the grouping cost. For a node that must begin / end a
  // group, it is positive if it would do so prematurely, or negative
  // if it would fit naturally into the schedule.
  GroupingCost = HazardRec.groupingCost(SU);

  // Check the resources cost for this SU.
  ResourcesCost = HazardRec.resourcesCost(SU);
}

bool SystemZPostRASchedStrategy::Candidate::
operator<(const Candidate &other) {

  // Check decoder grouping.
  if (GroupingCost < other.GroupingCost)
    return true;
  if (GroupingCost > other.GroupingCost)
    return false;

  // Compare the use of resources.
  if (ResourcesCost < other.ResourcesCost)
    return true;
  if (ResourcesCost > other.ResourcesCost)
    return false;

  // Higher SU is otherwise generally better.
  if (SU->getHeight() > other.SU->getHeight())
    return true;
  if (SU->getHeight() < other.SU->getHeight())
    return false;

  // If all same, fall back to original order.
  if (SU->NodeNum < other.SU->NodeNum)
    return true;

  return false;
}

void SystemZPostRASchedStrategy::schedNode(SUnit *SU, bool IsTopNode) {
  LLVM_DEBUG(dbgs() << "** Scheduling SU(" << SU->NodeNum << ") ";
             if (Available.size() == 1) dbgs() << "(only one) ";
             Candidate c(SU, *HazardRec); c.dumpCosts(); dbgs() << "\n";);

  // Remove SU from Available set and update HazardRec.
  Available.erase(SU);
  HazardRec->EmitInstruction(SU);
}

void SystemZPostRASchedStrategy::releaseTopNode(SUnit *SU) {
  // Set isScheduleHigh flag on all SUs that we want to consider first in
  // pickNode().
  const MCSchedClassDesc *SC = HazardRec->getSchedClass(SU);
  bool AffectsGrouping = (SC->isValid() && (SC->BeginGroup || SC->EndGroup));
  SU->isScheduleHigh = (AffectsGrouping || SU->isUnbuffered);

  // Put all released SUs in the Available set.
  Available.insert(SU);
}
