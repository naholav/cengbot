// Response/request shapes of the admin REST API (src/admin_rest_api.py).
// The backend is the source of truth: Optional[...] fields there may be null
// here, and datetimes arrive as ISO strings (naive UTC, i.e. without "Z").

export interface RawDataItem {
  id: number;
  telegram_id: number;
  telegram_message_id?: number | null;
  username?: string | null;
  question: string;
  answer?: string | null;
  language?: string | null;
  like?: number | null;
  admin_approved: number;
  is_duplicate: boolean;
  duplicate_of_id?: number | null;
  similarity_score?: number | null;
  created_at?: string | null;
  answered_at?: string | null;
  message_thread_id?: number | null;
  likes?: number | null;
  dislikes?: number | null;
  total_votes?: number | null;
  vote_score?: number | null;
}

export interface TrainingDataItem {
  id: number;
  source_id?: number | null;
  question: string;
  answer: string;
  language?: string | null;
  created_at?: string | null;
  /** 1 = liked, -1 = disliked, null = no rating (from the source raw_data). */
  point?: number | null;
  is_answer_duplicate?: boolean | null;
  duplicate_answer_of_id?: number | null;
  answer_similarity_score?: number | null;
}

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface Stats {
  total_questions: number;
  approved_questions: number;
  liked_questions: number;
  disliked_questions: number;
  training_data_count: number;
  duplicate_count: number;
  /** Present in newer API versions. */
  unanswered_questions?: number;
  /** e.g. "16.7%" */
  approval_rate: string;
}

/** Max answer length accepted by PUT /raw-data/{id} (backend MAX_ANSWER_LENGTH). */
export const MAX_ANSWER_LENGTH = 20000;

export interface UpdateAnswerRequest {
  answer: string;
}

export interface UpdateAnswerResponse extends MutationResponse {
  /** True when the approved training copy was updated as well. */
  training_data_updated?: boolean;
}

/** Generic `{success, message}` body returned by PUT/DELETE endpoints. */
export interface MutationResponse {
  success: boolean;
  message: string;
}

export interface ApproveResponse {
  success: boolean;
  message: string;
  training_data_id?: number | null;
}

export interface LoginResponse {
  success: boolean;
  username: string;
}

export type DocumentType = 'documentation' | 'log' | 'script';

export interface DocumentInfo {
  path: string;
  name: string;
  type: DocumentType;
  size: number;
}
