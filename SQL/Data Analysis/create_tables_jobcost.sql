CREATE DATABASE IF NOT EXISTS construction_jobcost
    DEFAULT CHARACTER SET utf8mb4;
USE construction_jobcost;

SET FOREIGN_KEY_CHECKS = 0;
DROP TABLE IF EXISTS fact_job_cost;
DROP TABLE IF EXISTS fact_budget;
DROP TABLE IF EXISTS fact_change_order;
DROP TABLE IF EXISTS fact_billing;
DROP TABLE IF EXISTS raw_invoice_extract;
DROP TABLE IF EXISTS dim_date;
DROP TABLE IF EXISTS dim_job;
DROP TABLE IF EXISTS dim_cost_code;
DROP TABLE IF EXISTS dim_vendor;
SET FOREIGN_KEY_CHECKS = 1;


CREATE TABLE dim_date (
    date_key        INT UNSIGNED      NOT NULL,
    full_date       DATE              NOT NULL,
    `year`          SMALLINT UNSIGNED NOT NULL,
    `quarter`       TINYINT UNSIGNED  NOT NULL,
    year_quarter    CHAR(7)           NOT NULL,
    `month`         TINYINT UNSIGNED  NOT NULL,
    month_name      VARCHAR(9)        NOT NULL,
    `year_month`      CHAR(7)           NOT NULL,
    day_of_month    TINYINT UNSIGNED  NOT NULL,
    day_name        VARCHAR(9)        NOT NULL,
    is_weekend      TINYINT UNSIGNED  NOT NULL DEFAULT 0,

    CONSTRAINT pk_dim_date      PRIMARY KEY (date_key),
    CONSTRAINT uq_dim_date_full UNIQUE (full_date),

    CONSTRAINT chk_date_key_range CHECK (date_key BETWEEN 19000101 AND 99991231),
    CONSTRAINT chk_date_quarter   CHECK (`quarter`    BETWEEN 1 AND 4),
    CONSTRAINT chk_date_month     CHECK (`month`      BETWEEN 1 AND 12),
    CONSTRAINT chk_date_dom       CHECK (day_of_month BETWEEN 1 AND 31),
    CONSTRAINT chk_date_weekend   CHECK (is_weekend IN (0, 1)),

    INDEX ix_dim_date_year_month (`year_month`),
    INDEX ix_dim_date_year_qtr   (year_quarter)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4;


CREATE TABLE dim_job (
    job_key              SMALLINT UNSIGNED NOT NULL AUTO_INCREMENT,
    job_number           VARCHAR(20)   NOT NULL,
    job_name             VARCHAR(100)  NOT NULL,
    customer             VARCHAR(100)  NOT NULL,
    job_type             VARCHAR(30)   NOT NULL,
    project_manager      VARCHAR(50)   NOT NULL,
    contract_amount      DECIMAL(14,2) NOT NULL,
    start_date           DATE          NOT NULL,
    planned_end_date     DATE          NOT NULL,
    status               VARCHAR(20)   NOT NULL DEFAULT 'Not Started',
    estimated_total_cost DECIMAL(14,2) NOT NULL,

    CONSTRAINT pk_dim_job        PRIMARY KEY (job_key),
    CONSTRAINT uq_dim_job_number UNIQUE (job_number),

    CONSTRAINT chk_job_contract_nonneg CHECK (contract_amount      >= 0),
    CONSTRAINT chk_job_est_cost_nonneg CHECK (estimated_total_cost >= 0),
    CONSTRAINT chk_job_dates           CHECK (planned_end_date >= start_date),
    CONSTRAINT chk_job_status          CHECK (status IN
                                          ('Not Started','In Progress','Complete',
                                           'On Hold','Cancelled')),
    CONSTRAINT chk_job_type            CHECK (job_type IN
                                          ('Commercial TI','Commercial New','Multifamily',
                                           'Institutional','Industrial','Residential')),

    INDEX ix_dim_job_status (status),
    INDEX ix_dim_job_pm     (project_manager)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4;


CREATE TABLE dim_cost_code (
    cost_code_key   SMALLINT UNSIGNED NOT NULL AUTO_INCREMENT,
    cost_code       VARCHAR(10)       NOT NULL,
    description     VARCHAR(60)       NOT NULL,
    division        CHAR(2)           NOT NULL,
    division_name   VARCHAR(40)       NOT NULL,
    cost_type       VARCHAR(15)       NOT NULL,

    CONSTRAINT pk_dim_cost_code      PRIMARY KEY (cost_code_key),
    CONSTRAINT uq_dim_cost_code_code UNIQUE (cost_code),

    CONSTRAINT chk_cc_cost_type CHECK (cost_type IN
                                  ('Labor','Material','Subcontract','Equipment','Other')),
    CONSTRAINT chk_cc_division  CHECK (division REGEXP '^[0-9]{2}$'),

    INDEX ix_dim_cost_code_division (division),
    INDEX ix_dim_cost_code_type     (cost_type)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4;


CREATE TABLE dim_vendor (
    vendor_key      SMALLINT UNSIGNED NOT NULL AUTO_INCREMENT,
    vendor_name     VARCHAR(100)      NOT NULL,
    vendor_type     VARCHAR(15)       NOT NULL,
    tax_id          VARCHAR(20)       NULL,
    payment_terms   VARCHAR(20)       NOT NULL DEFAULT 'Net 30',

    CONSTRAINT pk_dim_vendor      PRIMARY KEY (vendor_key),
    CONSTRAINT uq_dim_vendor_name UNIQUE (vendor_name),
    CONSTRAINT uq_dim_vendor_tax  UNIQUE (tax_id),

    CONSTRAINT chk_vendor_type  CHECK (vendor_type IN
                                  ('Material','Subcontract','Equipment','Labor','Other')),
    CONSTRAINT chk_vendor_terms CHECK (payment_terms IN
                                  ('Net 15','Net 30','Net 45','Net 60','Due on Receipt')),

    INDEX ix_dim_vendor_type (vendor_type)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4;


CREATE TABLE fact_budget (
    budget_key      INT UNSIGNED      NOT NULL AUTO_INCREMENT,
    job_key         SMALLINT UNSIGNED NOT NULL,
    cost_code_key   SMALLINT UNSIGNED NOT NULL,
    budget_amount   DECIMAL(14,2)     NOT NULL,
    budget_hours    DECIMAL(10,2)     NULL,

    CONSTRAINT pk_fact_budget  PRIMARY KEY (budget_key),
    CONSTRAINT uq_budget_grain UNIQUE (job_key, cost_code_key),

    CONSTRAINT fk_budget_job       FOREIGN KEY (job_key)
        REFERENCES dim_job (job_key)             ON UPDATE RESTRICT ON DELETE RESTRICT,
    CONSTRAINT fk_budget_cost_code FOREIGN KEY (cost_code_key)
        REFERENCES dim_cost_code (cost_code_key) ON UPDATE RESTRICT ON DELETE RESTRICT,

    CONSTRAINT chk_budget_amount_nonneg CHECK (budget_amount >= 0),
    CONSTRAINT chk_budget_hours_nonneg  CHECK (budget_hours IS NULL OR budget_hours >= 0)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4;


CREATE TABLE fact_job_cost (
    cost_key        INT UNSIGNED      NOT NULL AUTO_INCREMENT,
    date_key        INT UNSIGNED      NOT NULL,
    job_key         SMALLINT UNSIGNED NOT NULL,
    cost_code_key   SMALLINT UNSIGNED NOT NULL,
    vendor_key      SMALLINT UNSIGNED NULL,
    cost_type       VARCHAR(15)       NOT NULL,
    amount          DECIMAL(14,2)     NOT NULL,
    hours           DECIMAL(10,2)     NULL,
    source_type     VARCHAR(20)       NOT NULL,
    source_doc      VARCHAR(30)       NOT NULL,

    CONSTRAINT pk_fact_job_cost PRIMARY KEY (cost_key),

    CONSTRAINT fk_cost_date      FOREIGN KEY (date_key)
        REFERENCES dim_date (date_key)           ON UPDATE RESTRICT ON DELETE RESTRICT,
    CONSTRAINT fk_cost_job       FOREIGN KEY (job_key)
        REFERENCES dim_job (job_key)             ON UPDATE RESTRICT ON DELETE RESTRICT,
    CONSTRAINT fk_cost_cost_code FOREIGN KEY (cost_code_key)
        REFERENCES dim_cost_code (cost_code_key) ON UPDATE RESTRICT ON DELETE RESTRICT,
    CONSTRAINT fk_cost_vendor    FOREIGN KEY (vendor_key)
        REFERENCES dim_vendor (vendor_key)       ON UPDATE RESTRICT ON DELETE RESTRICT,

    CONSTRAINT chk_cost_type         CHECK (cost_type IN
                                        ('Labor','Material','Subcontract','Equipment','Other')),
    CONSTRAINT chk_cost_source       CHECK (source_type IN
                                        ('AP Invoice','Timecard','Journal Entry','Credit Memo')),
    CONSTRAINT chk_cost_hours_nonneg CHECK (hours IS NULL OR hours >= 0),
    CONSTRAINT chk_cost_source_shape CHECK (
        (source_type =  'Timecard' AND vendor_key IS     NULL AND hours IS NOT NULL)
     OR (source_type <> 'Timecard' AND vendor_key IS NOT NULL AND hours IS     NULL)
    ),

    INDEX ix_cost_job_code (job_key, cost_code_key),
    INDEX ix_cost_date     (date_key),
    INDEX ix_cost_doc      (source_doc)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4;


CREATE TABLE fact_change_order (
    change_order_key SMALLINT UNSIGNED NOT NULL AUTO_INCREMENT,
    co_number        VARCHAR(20)       NOT NULL,
    job_key          SMALLINT UNSIGNED NOT NULL,
    date_key         INT UNSIGNED      NOT NULL,
    amount           DECIMAL(14,2)     NOT NULL,
    reason           VARCHAR(60)       NOT NULL,
    status           VARCHAR(15)       NOT NULL DEFAULT 'Pending',
    days_added       SMALLINT          NOT NULL DEFAULT 0,

    CONSTRAINT pk_fact_change_order PRIMARY KEY (change_order_key),

    CONSTRAINT fk_co_job  FOREIGN KEY (job_key)
        REFERENCES dim_job (job_key)   ON UPDATE RESTRICT ON DELETE RESTRICT,
    CONSTRAINT fk_co_date FOREIGN KEY (date_key)
        REFERENCES dim_date (date_key) ON UPDATE RESTRICT ON DELETE RESTRICT,

    CONSTRAINT chk_co_status     CHECK (status IN ('Approved','Pending','Rejected','Void')),
    CONSTRAINT chk_co_days_added CHECK (days_added >= 0),

    INDEX ix_co_number (co_number),
    INDEX ix_co_job    (job_key, status)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4;


CREATE TABLE fact_billing (
    billing_key             INT UNSIGNED      NOT NULL AUTO_INCREMENT,
    job_key                 SMALLINT UNSIGNED NOT NULL,
    date_key                INT UNSIGNED      NOT NULL,
    `year_month`              CHAR(7)           NOT NULL,
    revised_contract_amount DECIMAL(14,2)     NOT NULL,
    billed_to_date          DECIMAL(14,2)     NOT NULL,
    retainage_held          DECIMAL(14,2)     NOT NULL,

    CONSTRAINT pk_fact_billing  PRIMARY KEY (billing_key),
    CONSTRAINT uq_billing_grain UNIQUE (job_key, `year_month`),

    CONSTRAINT fk_billing_job  FOREIGN KEY (job_key)
        REFERENCES dim_job (job_key)   ON UPDATE RESTRICT ON DELETE RESTRICT,
    CONSTRAINT fk_billing_date FOREIGN KEY (date_key)
        REFERENCES dim_date (date_key) ON UPDATE RESTRICT ON DELETE RESTRICT,

    CONSTRAINT chk_billing_billed_nonneg CHECK (billed_to_date >= 0),
    CONSTRAINT chk_billing_retain_nonneg CHECK (retainage_held >= 0),
    CONSTRAINT chk_billing_retainage     CHECK (retainage_held <= billed_to_date),
    CONSTRAINT chk_billing_overbill      CHECK (billed_to_date <= revised_contract_amount),
    CONSTRAINT chk_billing_ym_format     CHECK (`year_month` REGEXP '^[0-9]{4}-[0-9]{2}$'),

    INDEX ix_billing_period (`year_month`)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4;


CREATE TABLE raw_invoice_extract (
    raw_id          INT UNSIGNED     NOT NULL AUTO_INCREMENT,
    invoice_number  VARCHAR(50)      NULL,
    vendor_name_raw VARCHAR(200)     NULL,
    invoice_date    VARCHAR(50)      NULL,
    job_number_raw  VARCHAR(50)      NULL,
    cost_code_raw   VARCHAR(50)      NULL,
    amount_raw      VARCHAR(50)      NULL,
    po_number       VARCHAR(50)      NULL,
    source_file     VARCHAR(260)     NULL,
    load_ts         TIMESTAMP        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_processed    TINYINT UNSIGNED NOT NULL DEFAULT 0,
    reject_reason   VARCHAR(255)     NULL,

    CONSTRAINT pk_raw_invoice PRIMARY KEY (raw_id),

    INDEX ix_raw_invoice_number (invoice_number),
    INDEX ix_raw_job_number     (job_number_raw),
    INDEX ix_raw_processed      (is_processed)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4;
