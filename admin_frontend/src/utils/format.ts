// Matches a trailing "Z" or "+03:00" / "-0500" style UTC offset.
const HAS_TIMEZONE = /(Z|[+-]\d{2}:?\d{2})$/i;

/**
 * Parses a backend timestamp. The API stores `datetime.utcnow()` values and
 * serialises them without a timezone ("2025-07-18T08:26:40.123"); JS would
 * read those as *local* time, so treat zone-less values as UTC.
 */
export const parseServerDate = (value: string | null | undefined): Date | null => {
  if (!value) {
    return null;
  }
  const trimmed = value.trim();
  const iso = trimmed.includes('T') ? trimmed : trimmed.replace(' ', 'T');
  // Date-only strings are already parsed as UTC by the spec.
  const needsZ = iso.includes('T') && !HAS_TIMEZONE.test(iso);
  const date = new Date(needsZ ? `${iso}Z` : iso);
  return Number.isNaN(date.getTime()) ? null : date;
};

export const formatDateTime = (
  value: string | null | undefined,
  locale: string = 'tr-TR',
  options: Intl.DateTimeFormatOptions = {
    day: '2-digit',
    month: '2-digit',
    year: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }
): string => {
  const date = parseServerDate(value);
  return date ? date.toLocaleString(locale, options) : '-';
};

export const serverDateToMillis = (value: string | null | undefined): number =>
  parseServerDate(value)?.getTime() ?? 0;

export const formatSize = (bytes: number): string => {
  if (!Number.isFinite(bytes) || bytes < 0) return '-';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
};
