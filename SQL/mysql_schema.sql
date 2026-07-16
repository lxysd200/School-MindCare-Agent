SET NAMES utf8mb4;

CREATE TABLE IF NOT EXISTS `user_accounts` (
    `id` INT NOT NULL AUTO_INCREMENT,
    `username` VARCHAR(64) NOT NULL,
    `display_name` VARCHAR(128) NOT NULL,
    `password_hash` VARCHAR(128) NOT NULL,
    `roles_csv` VARCHAR(256) NOT NULL DEFAULT 'ROLE_USER',
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uq_user_accounts_username` (`username`),
    KEY `ix_user_accounts_username` (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `chat_sessions` (
    `id` INT NOT NULL AUTO_INCREMENT,
    `public_id` VARCHAR(64) NOT NULL,
    `title` VARCHAR(160) NOT NULL,
    `user_id` INT NOT NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uq_chat_sessions_public_id` (`public_id`),
    KEY `ix_chat_sessions_public_id` (`public_id`),
    KEY `ix_chat_sessions_user_id` (`user_id`),
    CONSTRAINT `fk_chat_sessions_user_id_user_accounts`
        FOREIGN KEY (`user_id`) REFERENCES `user_accounts` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `chat_messages` (
    `id` INT NOT NULL AUTO_INCREMENT,
    `user_id` INT NOT NULL,
    `session_id` INT NOT NULL,
    `role` VARCHAR(32) NOT NULL,
    `content` TEXT NOT NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `ix_chat_messages_user_id` (`user_id`),
    KEY `ix_chat_messages_session_id` (`session_id`),
    CONSTRAINT `fk_chat_messages_user_id_user_accounts`
        FOREIGN KEY (`user_id`) REFERENCES `user_accounts` (`id`),
    CONSTRAINT `fk_chat_messages_session_id_chat_sessions`
        FOREIGN KEY (`session_id`) REFERENCES `chat_sessions` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `knowledge_chunks` (
    `id` INT NOT NULL AUTO_INCREMENT,
    `source` VARCHAR(256) NOT NULL,
    `source_index` INT NOT NULL,
    `content` TEXT NOT NULL,
    `embedding_json` TEXT NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `ix_knowledge_chunks_source` (`source`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `psychological_reports` (
    `id` INT NOT NULL AUTO_INCREMENT,
    `user_id` INT NOT NULL,
    `session_id` INT NOT NULL,
    `content` TEXT NOT NULL,
    `intent` VARCHAR(32) NOT NULL,
    `emotion` VARCHAR(32) NOT NULL,
    `emotion_score` DOUBLE NOT NULL,
    `risk_level` VARCHAR(32) NOT NULL,
    `confidence` DOUBLE NOT NULL,
    `summary` TEXT NOT NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `ix_psychological_reports_user_id` (`user_id`),
    KEY `ix_psychological_reports_session_id` (`session_id`),
    CONSTRAINT `fk_psychological_reports_user_id_user_accounts`
        FOREIGN KEY (`user_id`) REFERENCES `user_accounts` (`id`),
    CONSTRAINT `fk_psychological_reports_session_id_chat_sessions`
        FOREIGN KEY (`session_id`) REFERENCES `chat_sessions` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `risk_cases` (
    `id` INT NOT NULL AUTO_INCREMENT,
    `report_id` INT NOT NULL,
    `risk_level` VARCHAR(32) NOT NULL,
    `status` VARCHAR(32) NOT NULL,
    `owner` VARCHAR(128) NOT NULL DEFAULT 'unassigned',
    `summary` TEXT NOT NULL,
    `handoff_summary` TEXT NOT NULL,
    `acknowledged_by` VARCHAR(128) NULL,
    `acknowledged_at` DATETIME NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uq_risk_cases_report_id` (`report_id`),
    KEY `ix_risk_cases_report_id` (`report_id`),
    KEY `ix_risk_cases_risk_level` (`risk_level`),
    KEY `ix_risk_cases_status` (`status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `case_notes` (
    `id` INT NOT NULL AUTO_INCREMENT,
    `case_id` INT NOT NULL,
    `actor` VARCHAR(128) NOT NULL,
    `note` TEXT NOT NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `ix_case_notes_case_id` (`case_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `alert_records` (
    `id` INT NOT NULL AUTO_INCREMENT,
    `report_id` INT NOT NULL,
    `channel` VARCHAR(64) NOT NULL,
    `recipient` VARCHAR(256) NOT NULL,
    `status` VARCHAR(32) NOT NULL,
    `message` TEXT NOT NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `ix_alert_records_report_id` (`report_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `excel_records` (
    `id` INT NOT NULL AUTO_INCREMENT,
    `report_id` INT NOT NULL,
    `file_path` VARCHAR(512) NOT NULL,
    `status` VARCHAR(32) NOT NULL,
    `message` TEXT NOT NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `ix_excel_records_report_id` (`report_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `tool_jobs` (
    `id` INT NOT NULL AUTO_INCREMENT,
    `report_id` INT NOT NULL,
    `kind` VARCHAR(64) NOT NULL,
    `status` VARCHAR(32) NOT NULL,
    `attempts` INT NOT NULL DEFAULT 0,
    `max_attempts` INT NOT NULL DEFAULT 3,
    `depends_on_job_id` INT NULL,
    `run_after` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `last_error` TEXT NOT NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `ix_tool_jobs_report_id` (`report_id`),
    KEY `ix_tool_jobs_kind` (`kind`),
    KEY `ix_tool_jobs_status` (`status`),
    KEY `ix_tool_jobs_depends_on_job_id` (`depends_on_job_id`),
    KEY `ix_tool_jobs_run_after` (`run_after`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `dead_letter_records` (
    `id` INT NOT NULL AUTO_INCREMENT,
    `job_id` INT NULL,
    `report_id` INT NOT NULL,
    `kind` VARCHAR(64) NOT NULL,
    `reason` TEXT NOT NULL,
    `payload` TEXT NOT NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `ix_dead_letter_records_job_id` (`job_id`),
    KEY `ix_dead_letter_records_report_id` (`report_id`),
    KEY `ix_dead_letter_records_kind` (`kind`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `agent_run_traces` (
    `id` INT NOT NULL AUTO_INCREMENT,
    `user_id` INT NOT NULL,
    `session_id` INT NOT NULL,
    `report_id` INT NULL,
    `intent` VARCHAR(32) NOT NULL,
    `risk_level` VARCHAR(32) NOT NULL DEFAULT 'LOW',
    `original_input` TEXT NOT NULL,
    `sanitized_input` TEXT NOT NULL,
    `memory_brief` TEXT NOT NULL,
    `agent_steps_json` TEXT NOT NULL,
    `retrieved_knowledge_json` TEXT NOT NULL,
    `response_messages_json` TEXT NOT NULL,
    `assessment_json` TEXT NOT NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `ix_agent_run_traces_user_id` (`user_id`),
    KEY `ix_agent_run_traces_session_id` (`session_id`),
    KEY `ix_agent_run_traces_report_id` (`report_id`),
    KEY `ix_agent_run_traces_intent` (`intent`),
    KEY `ix_agent_run_traces_risk_level` (`risk_level`),
    CONSTRAINT `fk_agent_run_traces_user_id_user_accounts`
        FOREIGN KEY (`user_id`) REFERENCES `user_accounts` (`id`),
    CONSTRAINT `fk_agent_run_traces_session_id_chat_sessions`
        FOREIGN KEY (`session_id`) REFERENCES `chat_sessions` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `tool_audit_records` (
    `id` INT NOT NULL AUTO_INCREMENT,
    `job_id` INT NULL,
    `report_id` INT NULL,
    `tool_name` VARCHAR(64) NOT NULL,
    `policy` VARCHAR(128) NOT NULL DEFAULT '',
    `allowed` BOOLEAN NOT NULL DEFAULT TRUE,
    `status` VARCHAR(32) NOT NULL,
    `reason` TEXT NOT NULL,
    `payload` TEXT NOT NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `ix_tool_audit_records_job_id` (`job_id`),
    KEY `ix_tool_audit_records_report_id` (`report_id`),
    KEY `ix_tool_audit_records_tool_name` (`tool_name`),
    KEY `ix_tool_audit_records_status` (`status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
