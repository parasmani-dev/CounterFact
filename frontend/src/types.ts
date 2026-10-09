export interface Suite { id: string; family_count: number; synthetic: boolean }
export interface Profile { name: string; key_present: boolean; last_status: string | null }
export interface Budget { used: number; cap: number; error_counts?: Record<string, number> }
export interface RunSummary { id: string; status: string; fresh: boolean; created_at: string; live_count: number | null; cached_count: number | null; budget: Budget }
export interface Observation { id: string; slot: number; raw_text: string; status: string; model: string; provider: string; acquired_at: string; cached: boolean }
export interface Side { image_hash: string | null; question: string; expected_answer: string | number | boolean; observations: Observation[]; verdict: string; correct_count: number; cached_count: number; live_count: number }
export interface Case { case_id: string; run_id: string; question_type: string; original: Side; transformed: Side; pair_status: string; eligible: boolean; eligibility_reason: string | null; minimization_id: string | null }
export interface Run { id: string; status: string; fresh?: boolean; budget: Budget; attempts_used?: number; error_counts?: Record<string, number>; cached_count?: number; live_count?: number; message?: string; error?: { message: string }; progress?: { completed_families: number; total_families: number } }
export interface Step { sequence: number; phase: string; candidate_hash: string; valid: boolean; accepted: boolean; size?: number[]; reason?: string }
export interface Spec { chart: { categories: unknown[] } }
export interface Minimization { id: string; status: string; calls_used: number; cap: number; cache_hits: number; elapsed: number; validity_rejections: number; confirmed: boolean; terminal_reason: string | null; description?: string; side?: 'original' | 'transformed'; best_candidate: Spec; start_size?: number[]; end_size?: number[]; steps: Step[]; artifacts?: Record<string, string>; error?: { message: string } }
export interface Regression { id: string; confirmed: boolean; case: unknown }
