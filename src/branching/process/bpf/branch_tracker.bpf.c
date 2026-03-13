// SPDX-License-Identifier: GPL-2.0
/*
 * branch_tracker.bpf.c — BPF LSM program for branch process tracking.
 *
 * Hooks task_alloc, task_free, task_kill, and ptrace_access_check to
 * inescapably track all descendants of registered branch root PIDs.
 * Supports atomic fork-denial during branch teardown, cross-branch
 * signal isolation, and cross-branch ptrace denial.
 *
 * Compile:
 *   clang -g -O2 -target bpf -D__TARGET_ARCH_x86 \
 *         -I. -c branch_tracker.bpf.c -o branch_tracker.bpf.o
 */

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

/* pid -> branch_id.  Updated on every fork by the LSM hook. */
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 10240);
	__type(key, __u32);
	__type(value, __u64);
} branch_pids SEC(".maps");

/* branch_id -> state.  0 = running, 1 = killing (deny forks). */
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 256);
	__type(key, __u64);
	__type(value, __u32);
} branch_state SEC(".maps");

/*
 * LSM hook: task_alloc — fires on every fork/clone.
 *
 * If the parent is a tracked branch process, the child is automatically
 * added to the same branch.  If the branch is being torn down
 * (state == 1), new forks are denied with -EPERM.
 */
SEC("lsm/task_alloc")
int BPF_PROG(branch_task_alloc, struct task_struct *task,
	     unsigned long clone_flags)
{
	__u32 ppid = BPF_CORE_READ(task, real_parent, tgid);
	__u64 *bid = bpf_map_lookup_elem(&branch_pids, &ppid);

	if (!bid)
		return 0; /* not a branch process — allow */

	/* If the branch is being torn down, deny new forks. */
	__u32 *state = bpf_map_lookup_elem(&branch_state, bid);
	if (state && *state == 1)
		return -1; /* -EPERM */

	/* Track the new child under the same branch. */
	__u32 child_pid = BPF_CORE_READ(task, tgid);
	__u64 branch_id = *bid;

	bpf_map_update_elem(&branch_pids, &child_pid, &branch_id, BPF_ANY);
	return 0;
}

/*
 * LSM hook: task_kill — fires on signal delivery (kill, tkill, tgkill).
 *
 * Enforces cross-branch signal isolation:
 *   - Untracked process → any target: allowed (parent/orchestrator)
 *   - Tracked process → same branch: allowed
 *   - Tracked process → different branch or untracked: denied (-EPERM)
 *
 * This prevents a child in one branch from killing processes in
 * another branch or the parent orchestrator.
 */
SEC("lsm/task_kill")
int BPF_PROG(branch_task_kill, struct task_struct *target,
	     struct kernel_siginfo *info, int sig,
	     const struct cred *cred)
{
	__u32 sender_pid = bpf_get_current_pid_tgid() >> 32;
	__u64 *sender_bid = bpf_map_lookup_elem(&branch_pids, &sender_pid);

	/* Untracked sender (parent/orchestrator) — always allow. */
	if (!sender_bid)
		return 0;

	__u32 target_pid = BPF_CORE_READ(target, tgid);
	__u64 *target_bid = bpf_map_lookup_elem(&branch_pids, &target_pid);

	/* Target is not tracked — deny (protects parent). */
	if (!target_bid)
		return -1; /* -EPERM */

	/* Both tracked — allow only within the same branch. */
	if (*sender_bid != *target_bid)
		return -1; /* -EPERM */

	return 0;
}

/*
 * LSM hook: ptrace_access_check — fires on ptrace attach/peek/poke.
 *
 * Same boundary logic as task_kill: a tracked process can only ptrace
 * processes within its own branch.  Untracked processes (the parent
 * orchestrator) are unrestricted.
 *
 * Without this, a malicious child could ptrace the parent or a sibling
 * branch process and read/write its memory, bypassing all other
 * isolation.
 */
SEC("lsm/ptrace_access_check")
int BPF_PROG(branch_ptrace_access_check, struct task_struct *child,
	     unsigned int mode)
{
	__u32 tracer_pid = bpf_get_current_pid_tgid() >> 32;
	__u64 *tracer_bid = bpf_map_lookup_elem(&branch_pids, &tracer_pid);

	/* Untracked tracer (parent/orchestrator) — always allow. */
	if (!tracer_bid)
		return 0;

	__u32 target_pid = BPF_CORE_READ(child, tgid);
	__u64 *target_bid = bpf_map_lookup_elem(&branch_pids, &target_pid);

	/* Target is not tracked — deny (protects parent). */
	if (!target_bid)
		return -1; /* -EPERM */

	/* Both tracked — allow only within the same branch. */
	if (*tracer_bid != *target_bid)
		return -1; /* -EPERM */

	return 0;
}

/*
 * LSM hook: task_free — fires when a process exits.
 *
 * Removes the process from tracking so the map stays clean.
 */
SEC("lsm/task_free")
int BPF_PROG(branch_task_free, struct task_struct *task)
{
	__u32 pid = BPF_CORE_READ(task, tgid);

	bpf_map_delete_elem(&branch_pids, &pid);
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
