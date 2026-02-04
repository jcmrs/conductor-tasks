import OpenAI from 'openai';
import process from 'process';
import { ErrorHandler, ErrorCategory, ErrorSeverity, TaskError } from '../../core/errorHandler.js';
const errorHandler = ErrorHandler.getInstance();
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
export class ZaiClient {
    constructor(model, apiKey, baseURL) {
        this.maxRetries = 3;
        this.apiKey = apiKey || process.env.ZAI_API_KEY;
        this.baseURL = baseURL || process.env.ZAI_BASE_URL || 'https://api.z.ai/api/coding/paas/v4';
        if (!this.apiKey) {
            throw new Error('API key is required for Z.ai client (either passed or via ZAI_API_KEY env var)');
        }
        this.client = new OpenAI({
            apiKey: this.apiKey,
            baseURL: this.baseURL,
        });
        this.model = model || process.env.ZAI_MODEL || 'glm-4.7';
        if (process.env.LLM_MAX_RETRIES) {
            this.maxRetries = parseInt(process.env.LLM_MAX_RETRIES, 10);
        }
    }
    async complete(options) {
        const { prompt, maxTokens = 4000, temperature = 0.7, topP, presencePenalty, frequencyPenalty, stream, onStreamUpdate, systemPrompt } = options;
        const params = {
            model: this.model,
            messages: systemPrompt
                ? [
                    { role: 'system', content: systemPrompt },
                    { role: 'user', content: prompt }
                ]
                : [{ role: 'user', content: prompt }],
            max_tokens: maxTokens,
            temperature: temperature,
            top_p: topP,
            presence_penalty: presencePenalty,
            frequency_penalty: frequencyPenalty,
            stream: stream,
        };
        let retryCount = 0;
        let lastError = null;
        while (retryCount <= this.maxRetries) {
            try {
                if (retryCount > 0) {
                    const backoffMs = Math.min(1000 * Math.pow(2, retryCount - 1), 10000);
                    await new Promise(resolve => setTimeout(resolve, backoffMs));
                    errorHandler.handleError(new TaskError(`Z.ai API error (retry ${retryCount}/${this.maxRetries}): ${lastError instanceof Error ? lastError.message : String(lastError)}`, ErrorCategory.LLM, ErrorSeverity.WARNING, { operation: 'zai-complete', additionalInfo: { retry: retryCount } }, lastError instanceof Error ? lastError : undefined), true);
                }
                if (stream && onStreamUpdate) {
                    const stream = await this.client.chat.completions.create({
                        ...params,
                        stream: true,
                    });
                    let fullResponse = '';
                    let finishReason = null;
                    for await (const chunk of stream) {
                        const content = chunk.choices[0]?.delta?.content || '';
                        if (content) {
                            fullResponse += content;
                            onStreamUpdate(content);
                        }
                        if (chunk.choices[0]?.finish_reason) {
                            finishReason = chunk.choices[0].finish_reason;
                        }
                    }
                    return {
                        text: fullResponse,
                        usage: null,
                        model: this.model,
                        finishReason: finishReason || undefined,
                    };
                }
                else {
                    const response = await this.client.chat.completions.create({
                        ...params,
                        stream: false,
                    });
                    const text = response.choices[0]?.message?.content || '';
                    const usage = response.usage
                        ? {
                            promptTokens: response.usage.prompt_tokens,
                            completionTokens: response.usage.completion_tokens,
                            totalTokens: response.usage.total_tokens,
                        }
                        : null;
                    return {
                        text: text,
                        usage: usage,
                        model: response.model,
                        finishReason: response.choices[0]?.finish_reason || undefined,
                    };
                }
            }
            catch (error) {
                lastError = error;
                const isRetryable = error instanceof Error &&
                    (error.message.includes('network') ||
                        error.message.includes('timeout') ||
                        error.message.includes('rate') ||
                        error.message.includes('limit') ||
                        error.message.includes('429') ||
                        error.message.includes('500') ||
                        error.message.includes('502') ||
                        error.message.includes('503') ||
                        error.message.includes('504'));
                if (isRetryable && retryCount < this.maxRetries) {
                    retryCount++;
                    continue;
                }
                errorHandler.handleError(new TaskError(`Z.ai API error: ${error instanceof Error ? error.message : String(error)}`, ErrorCategory.LLM, ErrorSeverity.ERROR, { operation: 'zai-complete' }, error instanceof Error ? error : undefined));
                throw new TaskError(`Z.ai API error: ${error instanceof Error ? error.message : String(error)}`, ErrorCategory.LLM, ErrorSeverity.ERROR, { operation: 'zai-complete' }, error instanceof Error ? error : undefined);
            }
        }
        errorHandler.handleError(new TaskError(`Z.ai API error (after ${this.maxRetries} retries): ${lastError instanceof Error ? lastError.message : String(lastError)}`, ErrorCategory.LLM, ErrorSeverity.ERROR, { operation: 'zai-complete', additionalInfo: { maxRetriesExceeded: true } }, lastError instanceof Error ? lastError : undefined));
        throw new TaskError(`Z.ai API error (after ${this.maxRetries} retries): ${lastError instanceof Error ? lastError.message : String(lastError)}`, ErrorCategory.LLM, ErrorSeverity.ERROR, { operation: 'zai-complete', additionalInfo: { maxRetriesExceeded: true } }, lastError instanceof Error ? lastError : undefined);
    }
    getProviderName() {
        return 'Z.ai';
    }
    getModelName() {
        return this.model;
    }
}
//# sourceMappingURL=zai.js.map