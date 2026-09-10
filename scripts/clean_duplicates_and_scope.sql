-- ==============================================================================
-- Devalign: Deduplication & Scope Cleanup for job_offers
-- ==============================================================================

-- 1. Eliminate exact duplicate postings (same job_title, company, and full_description)
-- Keeping the oldest/first record by UUID and cascading to offer_skills
DELETE FROM public.job_offers jo
WHERE jo.job_offer_id NOT IN (
    SELECT MIN(job_offer_id::text)::uuid
    FROM public.job_offers
    GROUP BY job_title, company, full_description
);

-- 2. Clean empty or pure whitespace job titles by unlinking cluster_id
UPDATE public.job_offers
SET cluster_id = NULL
WHERE TRIM(job_title) = '' OR job_title IS NULL;

-- 3. Verify counts
SELECT 
    COUNT(*) AS total_offers,
    COUNT(cluster_id) AS clustered_offers,
    COUNT(*) - COUNT(cluster_id) AS unclustered_offers
FROM public.job_offers;
