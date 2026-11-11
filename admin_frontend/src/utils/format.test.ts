import { formatDateTime, formatSize, parseServerDate, serverDateToMillis } from './format';

describe('parseServerDate', () => {
  it('treats zone-less backend timestamps (datetime.utcnow) as UTC', () => {
    expect(parseServerDate('2025-11-06T07:19:04.738283')?.toISOString()).toBe('2025-11-06T07:19:04.738Z');
    expect(parseServerDate('2025-11-06 07:19:04')?.toISOString()).toBe('2025-11-06T07:19:04.000Z');
  });

  it('respects explicit timezones', () => {
    expect(parseServerDate('2025-11-06T07:19:04Z')?.toISOString()).toBe('2025-11-06T07:19:04.000Z');
    expect(parseServerDate('2025-11-06T10:19:04+03:00')?.toISOString()).toBe('2025-11-06T07:19:04.000Z');
  });

  it('returns null for empty or invalid input', () => {
    expect(parseServerDate(null)).toBeNull();
    expect(parseServerDate(undefined)).toBeNull();
    expect(parseServerDate('')).toBeNull();
    expect(parseServerDate('not a date')).toBeNull();
  });
});

describe('formatDateTime', () => {
  it('formats in the given time zone from a UTC timestamp', () => {
    const text = formatDateTime('2025-11-06T07:19:04', 'en-GB', {
      timeZone: 'Europe/Istanbul',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    });
    expect(text).toBe('10:19');
  });

  it('shows a dash for missing values', () => {
    expect(formatDateTime(null)).toBe('-');
    expect(formatDateTime('garbage')).toBe('-');
  });
});

describe('serverDateToMillis', () => {
  it('orders timestamps and maps invalid ones to 0', () => {
    expect(serverDateToMillis('2025-01-02T00:00:00')).toBeGreaterThan(serverDateToMillis('2025-01-01T00:00:00'));
    expect(serverDateToMillis(null)).toBe(0);
  });
});

describe('formatSize', () => {
  it('formats bytes, KB, MB and GB', () => {
    expect(formatSize(512)).toBe('512 B');
    expect(formatSize(2048)).toBe('2.0 KB');
    expect(formatSize(5 * 1024 * 1024)).toBe('5.0 MB');
    expect(formatSize(86 * 1024 * 1024 * 1024)).toBe('86.0 GB');
    expect(formatSize(-1)).toBe('-');
  });
});
