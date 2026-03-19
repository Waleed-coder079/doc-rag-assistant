# Test Documentation

## Overview

This document catalogs all test cases for the Voilo Backend API, providing comprehensive coverage across authentication, user management, conversations, languages, vocabulary, settings, chat menu, hint generation services, learning-moment correction services, worker services, and WebSocket functionality.

**Total Tests:** 630  
**Passing:** 630  
**Flaky (isolation issues):** 0  
**Failed:** 0  
**Skipped:** 2 (WebSocket integration tests - opt-in)

**Last Updated:** 2026-03-17

---

## Table of Contents

1. [Authentication Endpoints](#1-authentication-endpoints)
2. [User Management Endpoints](#2-user-management-endpoints)
3. [Conversation Endpoints](#3-conversation-endpoints)
4. [Language Endpoints](#4-language-endpoints)
5. [Vocabulary Endpoints](#5-vocabulary-endpoints)
6. [Conversation Service Tests](#6-conversation-service-tests)
7. [Flow 2 Worker Tests](#7-flow-2-worker-tests)
8. [Flow 2 Service Tests](#8-flow-2-service-tests)
9. [Status Transition Tests](#9-status-transition-tests)
10. [WebSocket Lifecycle Tests](#10-websocket-lifecycle-tests)
11. [Settings Endpoints](#11-settings-endpoints)
12. [Chat Menu Endpoints](#12-chat-menu-endpoints)
13. [Flow 1 LLM + TTS Service Tests](#13-flow-1-llm--tts-service-tests)
14. [Hint LLM Service Tests](#14-hint-llm-service-tests)
15. [Learning Moment LLM Service Tests](#15-learning-moment-llm-service-tests)
---

## Test Documentation Format

| Test Case ID | Test Scenario | Input | Expected Result | Type (Positive/Negative) | Status |

---

## 1. Authentication Endpoints

**Endpoint Base:** `/api/v1/auth/*`
**Test File:** `tests/api/v1/test_auth.py`

### OAuth Login Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_AUTH_001 | Valid OAuth login with Google | `{"provider": "google", "redirect_url": "https://api.voilo.ai/auth/callback"}` | 200 OK with auth URL | Positive | Passed |
| TC_AUTH_002 | Missing provider in OAuth login | `{"redirect_url": "https://example.com"}` | 422 Validation Error | Negative | Passed |
| TC_AUTH_003 | Invalid redirect URL format | `{"provider": "google", "redirect_url": "not-a-url"}` | 422 Validation Error | Negative | Passed |

### Password Login Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_AUTH_004 | Valid password login | `{"email": "user@example.com", "password": "ValidPass123!"}` | 200 OK or 401 (invalid credentials) | Positive | Passed |
| TC_AUTH_005 | Login with invalid email format | `{"email": "invalid-email", "password": "ValidPass123!"}` | 422 Validation Error | Negative | Passed |
| TC_AUTH_006 | User enumeration prevention | Non-existent user vs wrong password | Same generic error message | Positive | Passed |

### Password Reset Request Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_AUTH_007 | Password reset request | `{"email": "user@example.com"}` | 200 OK with success message | Positive | Passed |
| TC_AUTH_008 | Password reset with empty email | `{"email": ""}` | 422 Validation Error | Negative | Passed |
| TC_AUTH_009 | Password reset with invalid email | `{"email": "invalid"}` | 422 Validation Error | Negative | Passed |

### Reset Password Confirmation Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_AUTH_010 | Reset password with valid tokens | Valid access & refresh tokens | 200 OK | Positive | Passed |
| TC_AUTH_011 | Reset password with short access token | Token < 20 chars | 400 Bad Request | Negative | Passed |
| TC_AUTH_012 | Reset password with long access token | Token > 2048 chars | 400 Bad Request | Negative | Passed |
| TC_AUTH_013 | Reset password with invalid JWT format | Invalid JWT structure | 400 Bad Request | Negative | Passed |

### Service Availability Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_AUTH_014 | Service unavailable handling | Supabase client unavailable | 503 Service Unavailable | Negative | Passed |

---

## 2. User Management Endpoints

**Endpoint Base:** `/api/v1/user-profile`
**Test File:** `tests/api/v1/test_users.py`

### User Signup Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_USER_001 | Valid user signup | Complete signup data | 201 Created | Positive | Passed |
| TC_USER_002 | Signup missing first_name | All fields except first_name | 422 Validation Error | Negative | Passed |
| TC_USER_003 | Signup missing last_name | All fields except last_name | 422 Validation Error | Negative | Passed |
| TC_USER_004 | Signup missing email | All fields except email | 422 Validation Error | Negative | Passed |
| TC_USER_005 | Signup missing password | All fields except password | 422 Validation Error | Negative | Passed |
| TC_USER_006 | Signup with invalid email format | Email: "invalid-email" | 422 Validation Error | Negative | Passed |
| TC_USER_007 | Signup with empty email | Email: "" | 422 Validation Error | Negative | Passed |
| TC_USER_008 | Signup with weak password | Password: "weak" | 422 Validation Error | Negative | Passed |
| TC_USER_009 | Signup with password lacking uppercase | Password: "password123!" | 422 Validation Error | Negative | Passed |
| TC_USER_010 | Signup with password lacking number | Password: "Password!" | 422 Validation Error | Negative | Passed |
| TC_USER_011 | Signup with password too short | Password: "Pass1!" | 422 Validation Error | Negative | Passed |
| TC_USER_012 | Signup with password too long | Password: 129 chars | 422 Validation Error | Negative | Passed |
| TC_USER_013 | Signup with first_name too long | first_name: 51 chars | 422 Validation Error | Negative | Passed |
| TC_USER_014 | Signup with last_name too long | last_name: 51 chars | 422 Validation Error | Negative | Passed |

### Get User Profile Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_USER_015 | Get user profile (authenticated) | Valid auth token | 200 OK with user data | Positive | Passed |
| TC_USER_016 | Get user profile (unauthenticated) | No auth token | 401 Unauthorized | Negative | Passed |

### Update User Profile Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_USER_017 | Update username only | `{"username": "newusername"}` | 200 OK | Positive | Passed |
| TC_USER_018 | Update display_name only | `{"display_name": "John Doe"}` | 200 OK | Positive | Passed |
| TC_USER_019 | Update both username and display_name | Both fields | 200 OK | Positive | Passed |
| TC_USER_020 | Update with empty request | `{}` | 400 Bad Request | Negative | Passed |
| TC_USER_021 | Update with taken username | Existing username | 409 Conflict | Negative | Passed |
| TC_USER_022 | Update with invalid username | Username: "123" (starts with number) | 400 Bad Request | Negative | Passed |
| TC_USER_023 | Update with reserved username | Username: "admin" | 400 Bad Request | Negative | Passed |

### Delete Account Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_USER_024 | Delete account (authenticated) | Valid auth token | 200 OK | Positive | Passed |
| TC_USER_025 | Delete account (unauthenticated) | No auth token | 401 Unauthorized | Negative | Passed |

### Service Availability Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_USER_026 | Signup Supabase unavailable | Supabase client error | 503 Service Unavailable | Negative | Passed |

---

## 3. Conversation Endpoints

**Endpoint Base:** `/api/v1/conversation/*`
**Test File:** `tests/api/v1/test_conversation.py`

### Start Conversation Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_CONV_001 | Start conversation (authenticated) | Valid user_language_id | 200 OK with session ID | Positive | Passed |
| TC_CONV_002 | Start conversation (unauthenticated) | No auth token | 401 Unauthorized | Negative | Passed |
| TC_CONV_003 | Start conversation with missing ID | Empty request | 422 Validation Error | Negative | Passed |
| TC_CONV_004 | Start conversation with invalid ID | user_language_id: "abc" | 422 Validation Error | Negative | Passed |
| TC_CONV_005 | Start conversation with non-existent ID | user_language_id: 99999 | 404 Not Found | Negative | Passed |
| TC_CONV_006 | Start conversation without active subscription | Free user | 403 Forbidden | Negative | Passed |

### End Conversation Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_CONV_007 | End conversation (authenticated) | Valid session_id | 200 OK | Positive | Passed |
| TC_CONV_008 | End conversation (unauthenticated) | No auth token | 401 Unauthorized | Negative | Passed |
| TC_CONV_009 | End conversation with missing ID | Empty request | 422 Validation Error | Negative | Passed |
| TC_CONV_010 | End conversation for non-owned session | Another user's session | 403 Forbidden | Negative | Passed |

---

## 4. Language Endpoints

**Endpoint Base:** `/api/v1/languages` and `/api/v1/user_languages/*`  
**Test File:** `tests/api/v1/test_languages.py`  
**Total Tests:** 75

### Endpoint Coverage

- `GET /api/v1/languages`
- `POST /api/v1/user_languages`
- `GET /api/v1/user_languages/me`
- `GET /api/v1/user_languages/me/active`
- `PUT /api/v1/user_languages/me/active`
- `PUT /api/v1/user_languages/me/inactive`
- `GET /api/v1/user_languages/{user_language_id}`
- `PATCH /api/v1/user_languages/{user_language_id}`
- `PUT /api/v1/user_languages/me/native`
- `GET /api/v1/user_languages/me/native`

### Get All Languages Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_LANG_001 | Get languages success | GET `/languages` | 200 OK | Positive | Passed |
| TC_LANG_002 | Get languages returns list | GET `/languages` | JSON list response | Positive | Passed |
| TC_LANG_003 | Empty when no languages | GET `/languages` | 200 OK + empty list allowed | Positive | Passed |
| TC_LANG_004 | Handles DB errors gracefully | GET `/languages` | Safe handling (no crash) | Negative | Passed |
| TC_LANG_005 | No sensitive data in errors | GET `/languages` | No stack/db leak | Positive | Passed |
| TC_LANG_006 | Response structure check | GET `/languages` | Valid response schema | Positive | Passed |
| TC_LANG_007 | Public endpoint no auth | GET `/languages` | 200 without auth | Positive | Passed |

### Deterministic Router Behavior Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_LANG_008 | Returns active languages | Mock active language rows | 200 + active rows | Positive | Passed |
| TC_LANG_009 | Returns [] on DB error | Mock DB exception | 200 + `[]` | Negative | Passed |
| TC_LANG_010 | Get my languages success | GET `/user_languages/me` (authed) | 200 + user language list | Positive | Passed |
| TC_LANG_011 | Get my languages DB error | Mock DB exception | 500 | Negative | Passed |
| TC_LANG_012 | Invalid language on select | POST `/user_languages` invalid code | 400 | Negative | Passed |
| TC_LANG_013 | Create and activate language | POST `/user_languages` valid | 201 + active language | Positive | Passed |
| TC_LANG_014 | Active language not found | GET `/user_languages/me/active` | 404 | Negative | Passed |
| TC_LANG_015 | Set active not in list | PUT `/user_languages/me/active` | 404 | Negative | Passed |
| TC_LANG_016 | Set inactive already inactive | PUT `/user_languages/me/inactive` | 200 + ok | Positive | Passed |
| TC_LANG_017 | Get settings not found | GET `/user_languages/{id}` missing | 404 | Negative | Passed |
| TC_LANG_018 | Update invalid CEFR | PATCH `/user_languages/{id}` invalid CEFR | 422 | Negative | Passed |
| TC_LANG_019 | Force slang off below B1 | PATCH `/user_languages/{id}` | 200 + `allow_slang=False` | Positive | Passed |
| TC_LANG_020 | Set native language success | PUT `/user_languages/me/native` | 200 + updated native_lang | Positive | Passed |
| TC_LANG_021 | Get native language success | GET `/user_languages/me/native` | 200 + native_lang | Positive | Passed |

### Select User Language Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_LANG_022 | Select without auth | POST `/user_languages` | 401 | Negative | Passed |
| TC_LANG_023 | Select with invalid token | POST `/user_languages` + invalid bearer | 401 | Negative | Passed |
| TC_LANG_024 | Missing ISO code | POST `/user_languages` body missing field | 401 (auth first) | Negative | Passed |
| TC_LANG_025 | Empty ISO code | POST `/user_languages` empty value | 401 (auth first) | Negative | Passed |
| TC_LANG_026 | Invalid ISO code | POST `/user_languages` invalid code | 401/404 path | Negative | Passed |
| TC_LANG_027 | Valid ISO code structure | POST `/user_languages` valid code | 401 (without auth) | Positive | Passed |
| TC_LANG_028 | Valid CEFR level | POST with `cefr_level=B1` | 401 (without auth) | Positive | Passed |
| TC_LANG_029 | Unsupported CEFR | POST with unsupported CEFR | 401/400 path | Negative | Passed |
| TC_LANG_030 | No CEFR uses default | POST without CEFR | 401 (without auth) | Positive | Passed |
| TC_LANG_031 | Already selected language | POST duplicate language | 401/409 path | Negative | Passed |
| TC_LANG_032 | All optional fields | POST full payload | 401 (without auth) | Positive | Passed |
| TC_LANG_033 | Minimal payload | POST required-only payload | 401 (without auth) | Positive | Passed |
| TC_LANG_034 | DB error handling on select | POST select with DB failure path | 401/500 path | Negative | Passed |
| TC_LANG_035 | Rollback on duplicate | POST duplicate with rollback path | 401/409 path | Negative | Passed |
| TC_LANG_036 | Success response format | POST success schema check path | 401 (without auth) | Positive | Passed |

### Get User Languages Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_LANG_037 | Get user languages without auth | GET `/user_languages/me` | 401 | Negative | Passed |
| TC_LANG_038 | Get user languages invalid token | GET `/user_languages/me` invalid bearer | 401 | Negative | Passed |
| TC_LANG_039 | Get user languages success path | GET `/user_languages/me` | 401 (without auth) / 200 (authed) | Positive | Passed |
| TC_LANG_040 | Returns list format | GET `/user_languages/me` | 401/200 with list path | Positive | Passed |
| TC_LANG_041 | Empty list when no selections | GET `/user_languages/me` | 401/200 empty list path | Positive | Passed |
| TC_LANG_042 | Scoped to current user | GET `/user_languages/me` | Current user only | Positive | Passed |
| TC_LANG_043 | Handles DB errors | GET `/user_languages/me` | 401/500 path | Negative | Passed |
| TC_LANG_044 | No sensitive error data | GET `/user_languages/me` | No stack/db leak | Positive | Passed |

### Cross-Endpoint Security Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_LANG_045 | Protected endpoints require auth | POST `/user_languages`, GET `/user_languages/me` | 401/422 | Negative | Passed |
| TC_LANG_046 | No sensitive data in errors | Invalid requests | No traceback leak | Positive | Passed |
| TC_LANG_047 | Consistent auth errors | Multiple protected endpoints | Consistent 401 behavior | Positive | Passed |

### User Language Constraint Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_LANG_048 | Duplicate language constraint | POST duplicate language | 401/409 path | Negative | Passed |
| TC_LANG_049 | Duplicate error message quality | POST duplicate language | Clear duplicate message path | Negative | Passed |

### CEFR Validation Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_LANG_050 | Accept valid CEFR levels | A1, A2, B1, B2, C1, C2 | Valid path (auth dependent) | Positive | Passed |
| TC_LANG_051 | Reject invalid CEFR list entry | CEFR `Z1` | 401/400 validation path | Negative | Passed |

### Optional Fields Validation Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_LANG_052 | level_estimate optional | Omit level_estimate | Accepted path (auth dependent) | Positive | Passed |
| TC_LANG_053 | auto_difficulty optional | Omit auto_difficulty | Accepted path (auth dependent) | Positive | Passed |
| TC_LANG_054 | difficulty_bias optional | Omit difficulty_bias | Accepted path (auth dependent) | Positive | Passed |
| TC_LANG_055 | speaking_speed optional | Omit speaking_speed | Accepted path (auth dependent) | Positive | Passed |
| TC_LANG_056 | show_target_text optional | Omit show_target_text | Accepted path (auth dependent) | Positive | Passed |

### Public vs Private Endpoint Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_LANG_057 | `/languages` is public | GET `/languages` | 200 | Positive | Passed |
| TC_LANG_058 | `/user_languages/*` is private | POST/GET user language routes | 401 | Negative | Passed |

### ISO Code Format Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_LANG_059 | Valid 2-letter ISO codes | `en, es, fr, de, it, pt, ja, zh` | Public list works, private route auth-first | Positive | Passed |

### Transaction Safety Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_LANG_060 | Rollback on integrity error | Duplicate/select conflict path | Rollback path verified | Negative | Passed |
| TC_LANG_061 | Commit on success | Successful select path | Commit path verified | Positive | Passed |

### Error Message Consistency Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_LANG_062 | Language not found message | Invalid language code | Clear not-found message path | Negative | Passed |
| TC_LANG_063 | CEFR unsupported message | Invalid CEFR for language | Clear CEFR message path | Negative | Passed |
| TC_LANG_064 | Duplicate language message | Duplicate selection | Clear duplicate message path | Negative | Passed |

### Success Response Format Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_LANG_065 | Language list metadata | GET `/languages` | List + expected fields | Positive | Passed |
| TC_LANG_066 | Select response schema | POST `/user_languages` | UserLanguage response schema path | Positive | Passed |
| TC_LANG_067 | User languages list schema | GET `/user_languages/me` | List schema path | Positive | Passed |

### Default Values Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_LANG_068 | Default CEFR level | Omit `cefr_level` | Defaults applied path | Positive | Passed |
| TC_LANG_069 | Default auto_difficulty | Omit `auto_difficulty` | Defaults applied path | Positive | Passed |
| TC_LANG_070 | Default difficulty_bias | Omit `difficulty_bias` | Defaults applied path | Positive | Passed |
| TC_LANG_071 | Default allow_slang | Omit `allow_slang` | Defaults applied path | Positive | Passed |
| TC_LANG_072 | Default speaking_speed | Omit `speaking_speed` | Defaults applied path | Positive | Passed |
| TC_LANG_073 | Default show_target_text | Omit `show_target_text` | Defaults applied path | Positive | Passed |
| TC_LANG_074 | Default xp points | New selection | Defaults applied path | Positive | Passed |
| TC_LANG_075 | Default is_active | New selection | Defaults applied path | Positive | Passed |

---

## 5. Vocabulary Endpoints

**Endpoint Base:** `/api/v1/vocabulary/*`
**Test File:** `tests/api/v1/test_vocabulary.py`

### Get Vocabulary Tags Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_VOCAB_001 | Get vocabulary tags | GET /languages/es-MX/tags | 200 OK with tags | Positive | Passed |
| TC_VOCAB_002 | Get tags with invalid ISO code | GET /languages/invalid/tags | 422 Validation Error | Negative | Passed |

### Get Active Words Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_VOCAB_003 | Get active words | GET /languages/es-MX/words/active | 200 OK with active words | Positive | Passed |
| TC_VOCAB_004 | Get active words with CEFR filter | GET /languages/es-MX/words/active?cefr=B1 | 200 OK with filtered words | Positive | Passed |
| TC_VOCAB_005 | Get active words invalid ISO | GET /languages/invalid/words/active | 422 Validation Error | Negative | Passed |
| TC_VOCAB_006 | Get active words unauthenticated | No auth token | 401 Unauthorized | Negative | Passed |

### Patch Word Status Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_VOCAB_007 | Patch word status | PATCH /words/{id} with status | 200 OK | Positive | Passed |
| TC_VOCAB_008 | Patch word invalid status | status: "invalid_status" | 422 Validation Error | Negative | Passed |

### Browse Words Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_VOCAB_009 | Browse words | GET /languages/es-MX/words/browse | 200 OK with words | Positive | Passed |
| TC_VOCAB_010 | Browse words with pagination | ?page=1&limit=10 | 200 OK with pagination | Positive | Passed |
| TC_VOCAB_011 | Browse words unauthenticated | No auth token | 401 Unauthorized | Negative | Passed |

### Activate Words Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_VOCAB_012 | Activate words | POST /words/activate with word_ids | 200 OK | Positive | Passed |
| TC_VOCAB_013 | Activate words unauthenticated | No auth token | 401 Unauthorized | Negative | Passed |
| TC_VOCAB_014 | Activate words with empty list | word_ids: [] | 400 Bad Request | Negative | Passed |

---

## 6. Conversation Service Tests

**Test File:** `app/tests/test_conversation_service.py`

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_SVC_001 | Build conversation context | Valid user, language_id | Context with snapshot | Positive | Passed |
| TC_SVC_002 | Enqueue on final transcript | is_final=True | Flow2 task enqueued | Positive | Passed |
| TC_SVC_003 | Skip enqueue on non-final | is_final=False | No task enqueued | Positive | Passed |
| TC_SVC_004 | Select focus words | Snapshot with focus_words | Returns focus words | Positive | Passed |

---

## 7. Flow 2 Worker Tests

**Test File:** `app/tests/worker/test_flow2.py`

### Error Detection Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_FLOW2_001 | Error detection skips low STT quality | stt_quality: 0.3 | Returns empty errors | Positive | Passed |
| TC_FLOW2_002 | Error detection calls OpenAI | Valid user_text | Calls OpenAI API | Positive | Passed |
| TC_FLOW2_003 | Error detection handles invalid items | Invalid error items | Skips invalid items | Positive | Passed |
| TC_FLOW2_004 | Error detection handles OpenAI failure | OpenAI error | Returns empty errors | Positive | Passed |
| TC_FLOW2_005 | Error detection enqueues update | Valid errors | Enqueues learning update | Positive | Passed |

### Learning Update Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_FLOW2_006 | Learning update handles duplicate turn | Higher turn_id in Redis | Skips processing | Positive | Passed |
| TC_FLOW2_007 | Learning update processes new turn | New turn_id | Processes and updates | Positive | Passed |
| TC_FLOW2_008 | Learning update increments snapshot | New turn | Version incremented | Positive | Passed |
| TC_FLOW2_009 | Learning update sets last_turn | Valid turn_id | Sets Redis key | Positive | Passed |
| TC_FLOW2_010 | Learning update handles None turn_id | turn_id: None | Processes without check | Positive | Passed |

---

## 8. Flow 2 Service Tests

**Test File:** `app/tests/worker/test_flow2_service.py`

### Process Learning Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_SVC2_001 | Process learning with no errors | Empty errors list | Returns 0 updated | Positive | Passed |
| TC_SVC2_002 | Process learning updates existing status | Existing word status | Updates status | Positive | Passed |
| TC_SVC2_003 | Process learning creates new status | No existing status | Creates new status | Positive | Passed |
| TC_SVC2_004 | Process learning increments attempts | Valid error | attempt_count +1 | Positive | Passed |
| TC_SVC2_005 | Process learning increments errors | Valid error | error_count +1 | Positive | Passed |
| TC_SVC2_006 | Process learning increments specific errors | Tense error | tense_errors_count +1 | Positive | Passed |
| TC_SVC2_007 | Process learning ignores unknown errors | Unknown error type | Counter unchanged | Positive | Passed |
| TC_SVC2_008 | Process learning handles missing word | Unknown lemma | Skips word | Positive | Passed |

### Build Snapshot Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_SVC2_009 | Build snapshot selects max 8 words | Many focus words | Max 8 words selected | Positive | Passed |
| TC_SVC2_010 | Build snapshot assigns priorities | Valid words | Priority 1-8 assigned | Positive | Passed |
| TC_SVC2_011 | Build snapshot handles lesson mode | Lesson mode | Lesson words prioritized | Positive | Passed |

### CEFR Rank Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_SVC2_012 | CEFR rank valid levels | A1, A2, B1, B2, C1, C2 | Correct rank order | Positive | Passed |
| TC_SVC2_013 | CEFR rank case insensitive | "a1", "A1" | Same rank | Positive | Passed |
| TC_SVC2_014 | CEFR rank invalid levels | "X1", "Z99" | Returns None | Positive | Passed |

### Days Since Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_SVC2_015 | Within CEFR hard filter | Word CEFR <= User CEFR | Included | Positive | Passed |
| TC_SVC2_016 | Days since valid timestamp | Valid datetime | Correct days | Positive | Passed |
| TC_SVC2_017 | Days since handles None | timestamp: None | Returns None | Positive | Passed |

### Score Active Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_SVC2_018 | Score active never used | First time word | High priority score | Positive | Passed |
| TC_SVC2_019 | Score active old word | Last used >7 days ago | Medium priority score | Positive | Passed |
| TC_SVC2_020 | Score active recent word | Last used today | Low priority score | Positive | Passed |
| TC_SVC2_021 | Score active mistake bonus | Has error_streak | Increased score | Positive | Passed |

### Score Mistake Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_SVC2_022 | Score mistake high error count | error_count > 5 | High priority score | Positive | Passed |
| TC_SVC2_023 | Score mistake recent error | Recent last_error_at | Increased score | Positive | Passed |
| TC_SVC2_024 | Score mistake recent use penalty | Recently used | Decreased score | Positive | Passed |

### Score Lesson Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_SVC2_025 | Score lesson new word | Never practiced | High priority score | Positive | Passed |
| TC_SVC2_026 | Score lesson recently practiced | Practiced <3 days ago | Low priority score | Positive | Passed |
| TC_SVC2_027 | Score lesson old word | Practiced >7 days ago | Medium priority score | Positive | Passed |

---

## 9. Status Transition Tests

**Test File:** `app/tests/worker/test_status_transitions.py`

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| STATUS_001 | Active promotes to Learned | 5+ correct, no mistakes | Status: Learned | Positive | Passed |
| STATUS_002 | Learned promotes to Mastered | 10+ correct, no mistakes | Status: Mastered | Positive | Passed |
| STATUS_003 | Learned demotes to Active | New mistake | Status: Active | Positive | Passed |
| STATUS_004 | Mastered demotes to Active | New mistake | Status: Active | Positive | Passed |
| STATUS_005 | Calculates correct_day_streak | Practice today | Streak +1 | Positive | Passed |
| STATUS_006 | Resets correct_day_streak | Gap >2 days | Streak: 1 | Positive | Passed |
| STATUS_007 | Handles CEFR level filtering | User CEFR < Word CEFR | Skips promotion | Positive | Passed |

---

## 10. WebSocket Lifecycle Tests

**Test File:** `app/tests/test_lifecycle.py`

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| WS_001 | Connect and disconnect | Valid WebSocket | Connection succeeds | Positive | Passed |
| WS_002 | Multiple connections same user | Two connections | Old one kicked | Positive | Passed |
| WS_003 | Reconnect handling | Disconnect and reconnect | New connection active | Positive | Passed |
| WS_004 | Heartbeat ping-pong | Ping message | Pong response | Positive | Passed |

---

## 11. Settings Endpoints

**Endpoint Base:** `/api/v1/settings*`  
**Test File:** `tests/api/v1/test_settings.py`

### Get Settings Hub Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_SETTINGS_001 | Get settings hub (authenticated) | `GET /settings` | 200 OK with account, preferences, navigation | Positive | Passed |
| TC_SETTINGS_002 | Get settings hub user not found | `GET /settings` with missing user | 404 Not Found (`User not found`) | Negative | Passed |
| TC_SETTINGS_003 | Get settings without authentication | No auth token | 401/403 Unauthorized | Negative | Passed |
| TC_SETTINGS_004 | Get settings response structure | Valid request | Has account, preferences, navigation sections | Positive | Passed |
| TC_SETTINGS_005 | Get settings with null username | `username: None` | 200 OK with null username | Positive | Passed |
| TC_SETTINGS_006 | Get settings notifications disabled | `notifications_enabled: False` | 200 OK with false flag | Positive | Passed |

### Get Account Details Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_SETTINGS_007 | Get account info (authenticated) | `GET /settings/account` | 200 OK with email, username, joined_at | Positive | Passed |
| TC_SETTINGS_008 | Get account info user not found | `GET /settings/account` with missing user | 404 Not Found (`User not found`) | Negative | Passed |
| TC_SETTINGS_009 | Get account info without authentication | No auth token | 401/403 Unauthorized | Negative | Passed |
| TC_SETTINGS_010 | Get account info response structure | Valid request | Has email, username, joined_at fields | Positive | Passed |
| TC_SETTINGS_011 | Get account info with null username | `username: None` | 200 OK with null username | Positive | Passed |
| TC_SETTINGS_012 | Get account info joined_at format | Valid request | ISO 8601 datetime string with 'T' | Positive | Passed |

### Toggle Notifications Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_SETTINGS_013 | Enable notifications | `PATCH /settings/notifications` + `{"enabled": true}` | 200 OK + `{"ok": true}` | Positive | Passed |
| TC_SETTINGS_014 | Disable notifications | `PATCH /settings/notifications` + `{"enabled": false}` | 200 OK + `{"ok": true}` | Positive | Passed |
| TC_SETTINGS_015 | Toggle notifications user not found | `PATCH /settings/notifications` with missing user | 404 Not Found (`User not found`) | Negative | Passed |
| TC_SETTINGS_016 | Toggle notifications missing field | `PATCH /settings/notifications` + `{}` | 422 Validation Error | Negative | Passed |
| TC_SETTINGS_017 | Toggle notifications invalid type | `PATCH /settings/notifications` + `{"enabled": null}` | 422 Validation Error | Negative | Passed |
| TC_SETTINGS_018 | Toggle notifications without auth | No auth token | 401/403 Unauthorized | Negative | Passed |
| TC_SETTINGS_019 | Toggle notifications extra fields | Extra fields in request | 200 OK (fields ignored) | Positive | Passed |
| TC_SETTINGS_020 | Toggle notifications idempotency | Enable when already enabled | 200 OK (no change) | Positive | Passed |
| TC_SETTINGS_021 | Toggle notifications empty body | No request body | 422 Validation Error | Negative | Passed |

### Settings Security Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_SETTINGS_022 | All endpoints require auth | Unauthenticated requests to all endpoints | 401/403 on all endpoints | Negative | Passed |
| TC_SETTINGS_023 | No sensitive data in errors | Error responses | No database/stack traces leaked | Positive | Passed |
| TC_SETTINGS_024 | Returns only current user data | Authenticated request | Only current user's data returned | Positive | Passed |


---

## 12. Chat Menu Endpoints

**Endpoint Base:** `/api/v1/chat/*` & `/api/v1/user-words`  
**Test File:** `tests/api/v1/test_chat_menu.py`

### Chat Hints Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_CHAT_001 | Get chat hints (authenticated) | `POST /chat/hints` | 200 OK with reply_suggestions, useful_words | Positive | Passed |
| TC_CHAT_002 | Chat hints no active language | User without active language | 404 Not Found (`Active user language not found`) | Negative | Passed |
| TC_CHAT_003 | Chat hints without authentication | No auth token | 401/403 Unauthorized | Negative | Passed |

### Learning Moment Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_CHAT_004 | Get learning moment turns | `GET /chat/learning-moment?conversation_id=1&offset=0` | 200 OK with turns array | Positive | Passed |
| TC_CHAT_005 | Learning moment empty result | No turns for conversation | 200 OK with empty array | Positive | Passed |
| TC_CHAT_006 | Learning moment without auth | No auth token | 401/403 Unauthorized | Negative | Passed |
| TC_CHAT_007 | Learning moment missing conversation_id | No conversation_id param | 422 Validation Error | Negative | Passed |

### AI Audio Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_CHAT_008 | Get AI audio signed URL | `GET /chat/learning-moment/ai-audio?turn_id=1` | 200 OK with signed_url | Positive | Passed |
| TC_CHAT_009 | AI audio turn not found | Non-existent turn_id | 404 Not Found (`Audio not found for this turn`) | Negative | Passed |
| TC_CHAT_010 | AI audio no audio path | Turn without audio_storage_path | 404 Not Found | Negative | Passed |
| TC_CHAT_011 | AI audio without authentication | No auth token | 401/403 Unauthorized | Negative | Passed |

### Corrected Text Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_CHAT_012 | Get corrected text | `POST /chat/learning-moment/corrected-text` + `{"turn_id": 1}` | 200 OK with corrected_text | Positive | Passed |
| TC_CHAT_013 | Corrected text turn not found | Non-existent turn_id | 404 Not Found (`Turn not found`) | Negative | Passed |
| TC_CHAT_014 | Corrected text without auth | No auth token | 401/403 Unauthorized | Negative | Passed |
| TC_CHAT_015 | Corrected text missing turn_id | Empty request body | 422 Validation Error | Negative | Passed |

### Corrected Audio Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_CHAT_016 | Get corrected audio | `POST /chat/learning-moment/corrected-audio` + `{"corrected_text": "text"}` | 200 OK with audio/mpeg | Positive | Passed |
| TC_CHAT_017 | Corrected audio without auth | No auth token | 401/403 Unauthorized | Negative | Passed |
| TC_CHAT_018 | Corrected audio missing text | Empty request body | 422 Validation Error | Negative | Passed |

### User Words Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_CHAT_019 | Save word to custom words | `POST /user-words` + `{"word": "newword"}` | 200 OK with result: saved_to_custom_words | Positive | Passed |
| TC_CHAT_020 | Save word no active language | User without active language | 404 Not Found (`Active user language not found`) | Negative | Passed |
| TC_CHAT_021 | Save word without authentication | No auth token | 401/403 Unauthorized | Negative | Passed |
| TC_CHAT_022 | Save word missing word field | Empty request body | 422 Validation Error | Negative | Passed |

### Chat Menu Security Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_CHAT_023 | All chat endpoints require auth | Unauthenticated requests to all endpoints | 401/403 on all endpoints | Negative | Passed |
| TC_CHAT_024 | No sensitive data in errors | Error responses | No database/stack traces leaked | Positive | Passed |


---

## 13. Flow 1 LLM + TTS Service Tests

**Test File:** `app/tests/test_flow1_llm_tts_service.py`

### Build Conversation Messages Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_FLOW1_001 | Build empty history | `[]` | Returns empty messages | Positive | Passed |
| TC_FLOW1_002 | Build single turn | One turn with user+ai text | Returns 2 messages (user, assistant) | Positive | Passed |
| TC_FLOW1_003 | Build multiple turns | Multi-turn history | Preserves all turns in sequence | Positive | Passed |
| TC_FLOW1_004 | Preserve order | Ordered turns | Chronological order retained | Positive | Passed |
| TC_FLOW1_005 | Filter both empty | user_text="", ai_text="" | Empty pair skipped | Negative | Passed |
| TC_FLOW1_006 | Include empty user only | user_text="", ai_text="..." | Assistant message included | Positive | Passed |
| TC_FLOW1_007 | Include empty ai only | user_text="...", ai_text="" | User message included | Positive | Passed |
| TC_FLOW1_008 | Mixed valid/empty turns | Mixed truthy/falsy text | Per-field inclusion works | Positive | Passed |
| TC_FLOW1_009 | Missing user_text field | Partial turn object | Graceful handling/robustness | Negative | Passed |

### Generate AI Reply Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_FLOW1_010 | Generate reply success | Valid transcript | Returns AI reply + usage | Positive | Passed |
| TC_FLOW1_011 | Reply with conversation history | Prior messages included | Contextual response returned | Positive | Passed |
| TC_FLOW1_012 | Reply with focus words | focus_words list | Reply generated with focus context | Positive | Passed |
| TC_FLOW1_013 | Reply with CEFR level | cefr_level provided | Reply generated with CEFR context | Positive | Passed |
| TC_FLOW1_014 | Fallback language path | DB language resolution path | Reply still generated | Positive | Passed |
| TC_FLOW1_015 | OpenAI error handling | OpenAI raises exception | Fallback safe reply returned | Negative | Passed |
| TC_FLOW1_016 | Usage stats mapping | Response with usage fields | prompt/completion/total tokens mapped | Positive | Passed |
| TC_FLOW1_017 | Long transcript handling | Large transcript input | Reply generated without failure | Positive | Passed |

### TTS Synthesis Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_FLOW1_018 | TTS success | Normal text | Audio bytes returned | Positive | Passed |
| TC_FLOW1_019 | TTS custom voice | `voice="nova"` | Requested voice used | Positive | Passed |
| TC_FLOW1_020 | TTS custom speed | `speed=1.5` | Requested speed used | Positive | Passed |
| TC_FLOW1_021 | TTS long text | Long input text | Audio generated successfully | Positive | Passed |
| TC_FLOW1_022 | TTS without usage metadata | usage=None | Audio still returned | Positive | Passed |
| TC_FLOW1_023 | TTS API error | Provider exception | Exception handled/raised | Negative | Passed |
| TC_FLOW1_024 | TTS returns bytes | Standard call | Output type is bytes | Positive | Passed |
| TC_FLOW1_025 | TTS all supported voices | voice loop | Works for all configured voices | Positive | Passed |

### Handle Conversation Turn Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_FLOW1_026 | Handle turn success | Valid user transcript | Returns AI reply + audio payload | Positive | Passed |
| TC_FLOW1_027 | Handle turn with session ID | Existing session_id | Turn linked to session flow | Positive | Passed |
| TC_FLOW1_028 | Handle turn LLM failure | LLM error | Returns controlled error payload | Negative | Passed |
| TC_FLOW1_029 | Handle turn TTS failure | TTS error | Returns controlled error payload | Negative | Passed |
| TC_FLOW1_030 | Handle turn storage failure | Upload/storage error | Returns controlled error payload | Negative | Passed |

### Edge Cases & Performance

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_FLOW1_031 | Very long user transcript | ~6000 chars | Turn handled without crash | Positive | Passed |
| TC_FLOW1_032 | Special characters transcript | Unicode + emoji text | Turn handled correctly | Positive | Passed |
| TC_FLOW1_033 | Message role alternation | Multi-turn conversation | Valid user/assistant role structure | Positive | Passed |
| TC_FLOW1_034 | Sequence integrity | Multi-turn content | Content sequence preserved | Positive | Passed |
| TC_FLOW1_035 | Turn handling time limit | Mocked full flow | Completes under threshold | Positive | Passed |
| TC_FLOW1_036 | Memory-efficient audio handling | Large audio bytes | Correct bytes length/handling | Positive | Passed |

---

## 14. Hint LLM Service Tests

**Test File:** `app/tests/test_hint_llm.py`

### Word Extraction Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_HINT_001 | Remove parenthetical meaning | `"comer (to eat)"` | Returns `"comer"` | Positive | Passed |
| TC_HINT_002 | Keep plain word | `"hablar"` | Returns unchanged word | Positive | Passed |

### Language Code Normalization Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_HINT_003 | Normalize base code | `"es"` | `"es"` | Positive | Passed |
| TC_HINT_004 | Normalize regional code | `"es-mx"` | `"es"` | Positive | Passed |
| TC_HINT_005 | Normalize uppercase regional code | `"EN-US"` | `"en"` | Positive | Passed |
| TC_HINT_006 | Normalize None | `None` | `None` | Positive | Passed |
| TC_HINT_007 | Invalid format handling | `"Spanish (Spain)"` | Returns `None` | Negative | Passed |

### Translation Helper Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_HINT_008 | Invalid target language | target=`"Spanish (Spain)"` | Raises `ValueError` | Negative | Passed |
| TC_HINT_009 | Skip translation when same language | source=`"en-us"`, target=`"en"` | Original items returned | Positive | Passed |
| TC_HINT_010 | Translate items success | valid item list | Returns translated list | Positive | Passed |
| TC_HINT_011 | Partial translation failure fallback | one item translation fails | Failed item kept original | Negative | Passed |

### Hint Generation Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_HINT_012 | Generate hints success structure | valid context + categories | reply_suggestions + useful_words structure | Positive | Passed |
| TC_HINT_013 | Missing required languages | missing `native_lang` or `target_language` | Raises `ValueError` | Negative | Passed |
| TC_HINT_014 | Invalid language codes in context | malformed language values | Raises `ValueError` | Negative | Passed |
| TC_HINT_015 | Enforce output limits | extra sentences/words from LLM | max 3 hints and max 6 words | Positive | Passed |
| TC_HINT_016 | Clean useful words before translation | words with parenthetical text | Cleans words then translates | Positive | Passed |

---

## 15. Learning Moment LLM Service Tests

**Test File:** `app/tests/test_learning_llm.py`

### Corrected Sentence Generation Tests

| Test Case ID | Test Scenario | Input | Expected Result | Type | Status |
|--------------|--------------|-------|-----------------|------|--------|
| TC_LM_001 | Generate corrected sentence success | Valid user text + error summary | Returns corrected text | Positive | Passed |
| TC_LM_002 | Strips whitespace from LLM output | LLM returns padded content | Returns trimmed text | Positive | Passed |
| TC_LM_003 | Empty LLM content fallback | LLM returns blank string | Returns clear-sentence fallback | Negative | Passed |
| TC_LM_004 | Exception safe fallback | OpenAI exception | Returns safe fallback response | Negative | Passed |
| TC_LM_005 | Prompt and params correctness | Valid request | Uses expected model/temperature/max_tokens/messages | Positive | Passed |
| TC_LM_006 | None content fallback | LLM returns `None` content | Returns clear-sentence fallback | Negative | Passed |
| TC_LM_007 | Missing usage robustness | Response without `usage` | Returns corrected text successfully | Positive | Passed |
| TC_LM_008 | Message role structure | Generated messages | System + user roles in order | Positive | Passed |
| TC_LM_009 | Error summary included | Error summary payload | Included in user prompt | Positive | Passed |
| TC_LM_010 | Language label prompt (es-mx) | code/name = es-mx/Spanish | Prompt contains `Spanish (es-mx)` | Positive | Passed |
| TC_LM_011 | Language label prompt (fr) | code/name = fr/French | Prompt contains `French (fr)` | Positive | Passed |
| TC_LM_012 | Language label prompt (pt-br) | code/name = pt-br/Portuguese | Prompt contains `Portuguese (pt-br)` | Positive | Passed |

---

## Test Status Legend

| Status | Description |
|--------|-------------|
| **Passed** | Test passes consistently in full suite |
| **Skipped** | Test is opt-in or requires special environment (2 tests) |
| **Flaky** | Test passes individually but may have isolation issues (0 tests) |
| **Failed** | Test consistently fails (0 tests) |

---

<!-- ...existing code... -->
## Execution Commands

```bash
# Run all tests
uv run pytest app/tests/ tests/ -v

# Flow 1 LLM + TTS tests
uv run pytest app/tests/test_flow1_llm_tts_service.py -v

# Hint LLM service tests
uv run pytest app/tests/test_hint_llm.py -v

# Learning Moment LLM service tests
uv run pytest app/tests/test_learning_llm.py -v

# Run specific test categories
uv run pytest tests/api/v1/test_auth.py -v
uv run pytest tests/api/v1/test_users.py -v
uv run pytest tests/api/v1/test_languages.py -v
uv run pytest tests/api/v1/test_vocabulary.py -v
uv run pytest tests/api/v1/test_conversation.py -v
uv run pytest tests/api/v1/test_settings.py -v
uv run pytest tests/api/v1/test_chat_menu.py -v
uv run pytest app/tests/worker/ -v
uv run pytest app/tests/test_lifecycle.py -v
```

### Service Layer Coverage

| Service | Tests | Status |
|---------|-------|--------|
| Conversation Service | 4 | Complete |
| Flow1 LLM + TTS Service | 36 | Complete |
| Hint LLM Service | 16 | Complete |
| Learning Moment LLM Service | 12 | Complete |
| Flow2 Worker | 10 | Complete |
| Flow2 Service | 27 | Complete |
| Status Transitions | 7 | Complete |
| WebSocket Lifecycle | 4 | Complete (2 opt-in integration tests) |

### Test Type Distribution

| Type | Count | Percentage |
|------|-------|------------|
| Positive Tests | 425 | 67% |
| Negative Tests | 203 | 32% |
| Skipped Tests | 2 | 1% |

---

## Security Testing Coverage

- **User Enumeration Prevention** - Generic error messages for non-existent users
- **Input Validation** - Comprehensive field validation on all endpoints
- **Authentication/Authorization** - Protected routes properly secured
- **SQL Injection Prevention** - Parameterized queries throughout
- **XSS Prevention** - Proper output sanitization
- **Password Security** - Complexity requirements, secure hashing
- **Audio/File Access** - Signed URLs with expiration for media files

---

## Known Issues

### Test Isolation Issues - RESOLVED ✅

All test isolation issues have been resolved. Previously flaky tests now pass consistently in the full test suite:

1. ✅ `test_conversation_service.py` - Conversation service context tests now properly isolated
2. ✅ `test_flow2.py` - Worker task tests now properly isolated  
3. ✅ `test_flow2_service.py` - Learning service tests now properly isolated
4. ✅ `test_flow1_llm_tts_service.py` - All Flow1 service tests now properly isolated

### WebSocket Integration Tests (Opt-in)

The following 2 tests are skipped by default as they require special environment setup:

1. `test_websocket_connect_disconnect_integration` - Requires `TEST_WEBSOCKET_INTEGRATION=1` and `TEST_JWT_TOKEN`
2. `test_websocket_reconnect_kicks_old_session_integration` - Requires `TEST_WEBSOCKET_INTEGRATION=1` and `TEST_JWT_TOKEN`

These are integration tests that test actual WebSocket connections and are opt-in for developers to run manually when needed.

---

## Additional Documentation

- **API Documentation:** See `API_DOCUMENTATION.md` for endpoint details
- **Architecture:** See `ARCHITECTURE.md` for system design
- **Deployment:** See `DEPLOYMENT.md` for deployment procedures

---

**Document Version:** 1.3  
**Last Review:** 2026-03-17  
**Maintained By:** Engineering Team
