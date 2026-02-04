import { LLMClient, LLMCompletionOptions, LLMCompletionResult } from '../types.js';
/**
 * Z.ai GLM Coding Plan Client
 *
 * Uses OpenAI-compatible API format with Z.ai's GLM models.
 * Documentation: https://docs.z.ai/devpack/overview
 *
 * Environment Variables:
 * - ZAI_API_KEY: Required API key from Z.ai
 * - ZAI_MODEL: Model to use (default: glm-4.7)
 * - ZAI_BASE_URL: Custom base URL (default: https://api.z.ai/api/coding/paas/v4)
 *
 * Available Models:
 * - glm-4.7 (recommended, latest)
 * - glm-4.6
 * - glm-4.5
 * - glm-4.5-air (faster, lighter)
 */
export declare class ZaiClient implements LLMClient {
    private client;
    private model;
    private maxRetries;
    private apiKey?;
    private baseURL;
    constructor(model?: string, apiKey?: string, baseURL?: string);
    complete(options: LLMCompletionOptions): Promise<LLMCompletionResult>;
    getProviderName(): string;
    getModelName(): string;
}
