export interface RawDataItem {
  id: number;
  question: string;
  answer: string;
  username?: string;
  telegram_id?: number;
  telegram_message_id?: number;
  created_at: string;
  like?: number;
  admin_approved: number;
  is_duplicate?: boolean;
  duplicate_of_id?: number;
  similarity_score?: number;
  likes?: number;
  dislikes?: number;
  total_votes?: number;
  vote_score?: number;
}

export interface TrainingDataItem {
  id: number;
  source_id?: number;
  question: string;
  answer: string;
  language?: string;
  created_at: string;
  point?: number;
  is_answer_duplicate?: boolean;
  duplicate_answer_of_id?: number;
  answer_similarity_score?: number;
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
  unanswered_questions: number;
  duplicate_count: number;
}

export interface UpdateAnswerRequest {
  answer: string;
}

export interface ApproveResponse {
  success: boolean;
  message: string;
  training_data_id?: number;
}