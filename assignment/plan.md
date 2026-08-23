# Dayflow — HRMS Build Plan

Stack: TypeScript, Node.js, React, Drizzle ORM, PostgreSQL (via Docker Compose), Zod.

---

## 1. Tech Stack & Rationale

| Layer | Choice | Notes |
|---|---|---|
| Language | TypeScript (strict mode) | end-to-end type safety |
| Frontend | React + Vite | fast dev server, SPA |
| Routing (FE) | React Router v6 | role-based route guards |
| State/data fetching | TanStack Query | caching, mutations, optimistic UI |
| Forms | React Hook Form + Zod resolver | shared validation schemas with backend |
| Styling | Tailwind CSS + shadcn/ui | quick, consistent component system |
| Backend | Node.js + Express (or Fastify) | REST API |
| Validation | Zod | shared schema package for FE/BE |
| ORM | Drizzle ORM | SQL-first, typed schema, migrations via drizzle-kit |
| DB | PostgreSQL 16 | run via Docker Compose |
| Auth | JWT (access + refresh) + bcrypt/argon2 | httpOnly cookies for refresh token |
| Email | Nodemailer + a transactional provider (Resend/SMTP) | verification, alerts |
| File storage | Local disk (dev) → S3-compatible (prod, e.g. MinIO/Cloudflare R2) | profile pictures, documents |
| Testing | Vitest + Supertest (API), React Testing Library (FE) | |
| Linting/formatting | ESLint + Prettier | |
| Containerization | Docker Compose (db, adminer, api, optionally web) | |
| CI | GitHub Actions | lint, typecheck, test, build |

---

## 2. Monorepo Structure

Use a pnpm workspace (or npm workspaces) to share Zod schemas/types between frontend and backend.

```
dayflow/
├── apps/
│   ├── api/                 # Express/Fastify backend
│   │   ├── src/
│   │   │   ├── db/
│   │   │   │   ├── schema/          # drizzle table definitions
│   │   │   │   │   ├── users.ts
│   │   │   │   │   ├── employees.ts
│   │   │   │   │   ├── attendance.ts
│   │   │   │   │   ├── leaves.ts
│   │   │   │   │   ├── payroll.ts
│   │   │   │   │   ├── documents.ts
│   │   │   │   │   ├── notifications.ts
│   │   │   │   │   └── index.ts
│   │   │   │   ├── migrations/      # drizzle-kit generated SQL
│   │   │   │   ├── client.ts        # drizzle db instance
│   │   │   │   └── seed.ts
│   │   │   ├── modules/
│   │   │   │   ├── auth/            # signup, signin, verify-email, refresh
│   │   │   │   ├── users/
│   │   │   │   ├── employees/
│   │   │   │   ├── attendance/
│   │   │   │   ├── leaves/
│   │   │   │   ├── payroll/
│   │   │   │   ├── dashboard/
│   │   │   │   └── notifications/
│   │   │   │       (each module: controller.ts, service.ts, routes.ts, validators.ts)
│   │   │   ├── middleware/
│   │   │   │   ├── auth.middleware.ts     # verify JWT
│   │   │   │   ├── rbac.middleware.ts     # role guard (admin/employee)
│   │   │   │   ├── error.middleware.ts
│   │   │   │   ├── rateLimiter.ts
│   │   │   │   └── validateRequest.ts     # zod middleware
│   │   │   ├── lib/
│   │   │   │   ├── jwt.ts
│   │   │   │   ├── password.ts (argon2/bcrypt)
│   │   │   │   ├── mailer.ts
│   │   │   │   ├── logger.ts (pino)
│   │   │   │   └── env.ts (zod-validated env)
│   │   │   ├── app.ts
│   │   │   └── server.ts
│   │   ├── drizzle.config.ts
│   │   ├── .env.example
│   │   ├── Dockerfile
│   │   └── package.json
│   └── web/                 # React frontend
│       ├── src/
│       │   ├── pages/
│       │   │   ├── auth/ (SignIn, SignUp, VerifyEmail)
│       │   │   ├── employee/ (Dashboard, Profile, Attendance, LeaveRequests)
│       │   │   └── admin/ (Dashboard, EmployeeList, AttendanceRecords, LeaveApprovals, Payroll, Reports)
│       │   ├── components/
│       │   │   ├── ui/ (shadcn primitives)
│       │   │   ├── layout/ (Sidebar, Topbar, DashboardCard)
│       │   │   └── shared/ (StatusBadge, DataTable, DateRangePicker)
│       │   ├── features/
│       │   │   ├── auth/ (hooks, api calls)
│       │   │   ├── attendance/
│       │   │   ├── leaves/
│       │   │   └── payroll/
│       │   ├── lib/
│       │   │   ├── apiClient.ts (axios/fetch wrapper)
│       │   │   ├── queryClient.ts
│       │   │   └── auth-context.tsx
│       │   ├── routes/
│       │   │   ├── ProtectedRoute.tsx
│       │   │   └── RoleRoute.tsx
│       │   ├── App.tsx
│       │   └── main.tsx
│       ├── Dockerfile
│       └── package.json
├── packages/
│   └── shared/
│       ├── src/
│       │   ├── schemas/     # Zod schemas: auth, employee, attendance, leave, payroll
│       │   ├── types/       # inferred TS types from schemas
│       │   └── constants/   # roles, leave types, attendance statuses
│       └── package.json
├── docker-compose.yml
├── docker-compose.override.yml (optional dev overrides)
├── .github/workflows/ci.yml
├── pnpm-workspace.yaml
├── package.json
└── README.md
```

---

## 3. Docker Compose Setup

```yaml
# docker-compose.yml
services:
  postgres:
    image: postgres:16-alpine
    container_name: dayflow-db
    restart: unless-stopped
    environment:
      POSTGRES_USER: dayflow
      POSTGRES_PASSWORD: dayflow_dev_pw
      POSTGRES_DB: dayflow
    ports:
      - "5432:5432"
    volumes:
      - dayflow_pg_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U dayflow"]
      interval: 5s
      timeout: 5s
      retries: 5

  adminer:
    image: adminer
    restart: unless-stopped
    ports:
      - "8081:8080"
    depends_on:
      - postgres

  api:
    build: ./apps/api
    env_file: ./apps/api/.env
    ports:
      - "4000:4000"
    depends_on:
      postgres:
        condition: service_healthy
    volumes:
      - ./apps/api:/app
      - /app/node_modules

volumes:
  dayflow_pg_data:
```

Local dev flow: `docker compose up -d postgres adminer` for DB only during early dev (run API/web via `pnpm dev` on host for fast HMR), later add `api`/`web` services for full containerized runs.

---

## 4. Database Schema (Drizzle)

Core tables derived from the SRS:

**users**
- id (uuid, pk)
- employee_id (unique, string) — used at signup
- email (unique, string)
- password_hash (string)
- role (enum: `admin`, `employee`)
- is_email_verified (boolean, default false)
- created_at, updated_at

**email_verification_tokens**
- id, user_id (fk), token, expires_at

**refresh_tokens**
- id, user_id (fk), token_hash, expires_at, revoked_at

**employee_profiles**
- id, user_id (fk, unique)
- full_name, phone, address, profile_picture_url
- department, designation, date_of_joining
- job_type (full-time/part-time/contract)
- manager_id (fk → users, nullable)

**salary_structures**
- id, employee_id (fk)
- basic, hra, allowances (jsonb), deductions (jsonb)
- ctc, effective_from
- updated_by (fk → users, admin)

**payroll_runs / payslips** (optional, phase 2)
- id, employee_id (fk), month, year, gross, net, generated_at, pdf_url

**documents**
- id, employee_id (fk), type (id_proof, offer_letter, etc.), file_url, uploaded_at

**attendance_records**
- id, employee_id (fk)
- date
- check_in_time, check_out_time
- status (enum: `present`, `absent`, `half_day`, `leave`)
- unique constraint (employee_id, date)

**leave_requests**
- id, employee_id (fk)
- leave_type (enum: `paid`, `sick`, `unpaid`)
- start_date, end_date
- remarks
- status (enum: `pending`, `approved`, `rejected`)
- reviewed_by (fk → users, nullable), reviewer_comment, reviewed_at
- created_at

**notifications**
- id, user_id (fk), title, message, is_read, type, created_at

All enums defined via `pgEnum` in Drizzle; all tables have `created_at`/`updated_at` timestamps with defaults; foreign keys use `onDelete: 'cascade'` where child data (e.g. tokens) should not orphan.

Migration workflow: `drizzle-kit generate` → review SQL in `migrations/` → `drizzle-kit migrate` (or a custom `migrate.ts` runner executed on API boot in dev / via CI step in prod).

---

## 5. Shared Validation (Zod) — `packages/shared`

Define once, use on both client and server:
- `signUpSchema`, `signInSchema`
- `updateProfileSchema` (employee-editable subset) vs `adminUpdateEmployeeSchema` (full)
- `checkInSchema`, `checkOutSchema`
- `applyLeaveSchema`, `reviewLeaveSchema`
- `updateSalarySchema`
- `env schema` (backend only) for `process.env` validation at boot

Backend applies these via a `validateRequest(schema)` Express middleware; frontend applies them via `zodResolver` in React Hook Form — guarantees identical rules everywhere.

---

## 6. API Design (REST, versioned `/api/v1`)

**Auth**
- `POST /auth/signup`
- `GET /auth/verify-email?token=`
- `POST /auth/signin`
- `POST /auth/refresh`
- `POST /auth/logout`

**Users/Profile**
- `GET /me`
- `PATCH /me/profile` (employee, limited fields)
- `GET /employees` (admin — list, filter, paginate)
- `GET /employees/:id` (admin)
- `PATCH /employees/:id` (admin — full edit)

**Attendance**
- `POST /attendance/check-in`
- `POST /attendance/check-out`
- `GET /attendance/me?range=daily|weekly`
- `GET /attendance?employeeId=&from=&to=` (admin, all employees)

**Leaves**
- `POST /leaves` (employee applies)
- `GET /leaves/me`
- `GET /leaves` (admin — all, filterable by status)
- `PATCH /leaves/:id/approve`
- `PATCH /leaves/:id/reject`

**Payroll**
- `GET /payroll/me` (read-only)
- `GET /payroll/:employeeId` (admin)
- `PUT /payroll/:employeeId` (admin — update salary structure)

**Dashboard/Reports**
- `GET /dashboard/employee-summary`
- `GET /dashboard/admin-summary`
- `GET /reports/attendance?from=&to=&format=csv|pdf`
- `GET /reports/salary-slip/:employeeId?month=&year=`

**Notifications**
- `GET /notifications`
- `PATCH /notifications/:id/read`

Every route: Zod-validated body/query, `auth.middleware` (JWT check), `rbac.middleware` for admin-only routes, centralized error handler returning consistent `{ error: { code, message } }` shape.

---

## 7. Auth & Security Details

- Passwords hashed with argon2id (preferred) or bcrypt (cost ≥ 12).
- Access token (JWT, ~15 min expiry) returned in response body; refresh token (opaque or JWT, ~7–30 days) stored in httpOnly, secure, sameSite=strict cookie, and hashed copy stored in `refresh_tokens` table for revocation.
- Email verification token: random UUID/hash, expires in 24h, resend endpoint with rate limiting.
- Rate limiting on `/auth/*` (e.g. `express-rate-limit`).
- RBAC middleware checks `req.user.role` against route requirement (`admin` only vs `any authenticated`).
- Input sanitation via Zod on every mutating route.
- Helmet for HTTP headers, CORS locked to frontend origin.
- Audit-style `reviewed_by`/`updated_by` fields for accountability on approvals and salary edits.

---

## 8. Frontend Architecture

- `AuthProvider` (context) holds user + access token in memory; silent refresh via TanStack Query on app load using the refresh cookie.
- `ProtectedRoute` — redirects unauthenticated users to sign-in.
- `RoleRoute` — restricts `/admin/*` to `role === 'admin'`.
- Dashboard cards (Profile, Attendance, Leave Requests, Logout) as reusable `DashboardCard` component, data-driven by role.
- `DataTable` shared component (sorting/pagination) reused for Employee List, Attendance Records, Leave Approvals.
- Forms: Sign Up, Sign In, Edit Profile, Apply Leave, Salary Edit — all React Hook Form + shared Zod schemas + shadcn form components.
- Notifications: polling or simple TanStack Query `refetchInterval`, upgrade path to WebSocket/SSE later.

---

## 9. Milestones / Build Order

1. **Bootstrap** — monorepo, workspaces, shared package, ESLint/Prettier, Docker Compose (postgres + adminer), env validation.
2. **DB layer** — Drizzle schema for users/profiles, drizzle-kit config, first migration, seed script (1 admin + a few employees).
3. **Auth module** — signup, email verification, signin, JWT + refresh flow, RBAC middleware. Build matching FE sign-up/sign-in pages.
4. **Profile management** — view/edit profile (employee + admin variants), file upload for profile picture/documents.
5. **Attendance** — check-in/check-out endpoints, daily/weekly views, admin all-employee view.
6. **Leave management** — apply, list, approve/reject, status propagation, notifications on status change.
7. **Payroll** — read-only employee view, admin salary CRUD, salary slip generation (PDF, phase 2).
8. **Dashboards & reports** — aggregate summary endpoints, analytics/reports page, CSV/PDF export.
9. **Notifications & email alerts** — leave decisions, attendance reminders.
10. **Hardening** — tests (unit + integration), rate limiting, logging, error tracking, CI pipeline, Dockerize api/web fully, deployment docs.

---

## 10. Testing Strategy

- **Unit**: services (business logic), Zod schemas, utility functions — Vitest.
- **Integration**: API routes against a real Postgres test container (or the same docker-compose db with a `dayflow_test` database) — Supertest + Vitest.
- **Frontend**: component tests (forms, protected routes) — React Testing Library.
- **E2E** (stretch): Playwright covering signup → verify → signin → apply leave → admin approve.

---

## 11. Environment Variables (`apps/api/.env.example`)

```
NODE_ENV=development
PORT=4000
DATABASE_URL=postgres://dayflow:dayflow_dev_pw@localhost:5432/dayflow
JWT_ACCESS_SECRET=change_me
JWT_REFRESH_SECRET=change_me
JWT_ACCESS_EXPIRES_IN=15m
JWT_REFRESH_EXPIRES_IN=7d
SMTP_HOST=
SMTP_PORT=
SMTP_USER=
SMTP_PASS=
MAIL_FROM=noreply@dayflow.app
CLIENT_ORIGIN=http://localhost:5173
FILE_STORAGE_DRIVER=local
```

Validate with a Zod `envSchema` and `envSchema.parse(process.env)` at server boot — fail fast on misconfiguration.

---

## 12. Open Items / Future Enhancements (from source doc, phase 2+)

- Analytics & reports dashboard (salary slips, attendance reports) — beyond MVP CRUD.
- Notification system beyond in-app (email/push alerts).
- Excalidraw diagram referenced in source doc should be reviewed for any UI/flow details not captured here: https://link.excalidraw.com/l/65VNwvy7c4X/58RLEJ4oOwh
- Payroll run automation / payslip PDF generation.
- Multi-manager approval chains, org-chart view.
