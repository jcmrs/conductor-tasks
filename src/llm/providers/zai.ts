import { LLMClient, LLMCompletionOptions, LLMCompletionResult, LLMUsage } from '../types.js';
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
 * - glm-4.7 (recommended, latest) - 200K context, 128K output
 * - glm-4.6
 * - glm-4.5
 * - glm-4.5-air (faster, lighter)
 *
 * Compatibility Features:
 * - JSON request detection: Auto-adjusts temperature and system prompt for reliable JSON output
 * - Thinking tag stripping: Removes GLM's <think> blocks from responses
 * - OpenAI-compatible: Uses standard chat completion format
 *
 * GLM-4.7 Capabilities:
 * - Function/tool calling (OpenAI format)
 * - Interleaved thinking mode
 * - Strong coding benchmarks (LiveCodeBench-v6: 84.9, SWE-bench: 73.8%)
 */
export class ZaiClient implements LLMClient {
  private client: OpenAI;
  private model: string;
  private maxRetries: number = 3;
  private apiKey?: string;
  private baseURL: string;

  constructor(model?: string, apiKey?: string, baseURL?: string) {
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

  async complete(options: LLMCompletionOptions): Promise<LLMCompletionResult> {
    const {
      prompt,
      maxTokens = 4000,
      temperature = 0.7,
      topP,
      presencePenalty,
      frequencyPenalty,
      stream,
      onStreamUpdate,
      systemPrompt
    } = options;

    // Detect JSON requests (matching Anthropic provider pattern for compatibility)
    const isJsonRequest = systemPrompt?.includes('JSON') ||
                          systemPrompt?.includes('json') ||
                          prompt?.includes('JSON') ||
                          prompt?.includes('json');

    let effectiveSystemPrompt = systemPrompt;
    let effectiveTemperature = temperature;

    if (isJsonRequest) {
      // Enhance system prompt for reliable JSON output
      if (!effectiveSystemPrompt) {
        effectiveSystemPrompt = "CRITICAL: You are a pure JSON response system. You MUST ONLY output valid JSON with ABSOLUTELY NOTHING before or after it. ANY text outside the JSON will cause system failure.";
      } else if (!effectiveSystemPrompt.toLowerCase().includes('json-only') && !effectiveSystemPrompt.toLowerCase().includes('pure json')) {
        effectiveSystemPrompt = "CRITICAL: Output ONLY valid JSON with NOTHING else. ANY text outside the JSON will cause system failure.\n\n" + effectiveSystemPrompt;
      }

      // Lower temperature for more deterministic JSON output
      if (effectiveTemperature > 0.1) {
        effectiveTemperature = 0.05;
      }
    }

    const params: OpenAI.Chat.ChatCompletionCreateParams = {
      model: this.model,
      messages: effectiveSystemPrompt
        ? [
            { role: 'system', content: effectiveSystemPrompt },
            { role: 'user', content: prompt }
          ]
        : [{ role: 'user', content: prompt }],
      max_tokens: maxTokens,
      temperature: effectiveTemperature,
      top_p: topP,
      presence_penalty: presencePenalty,
      frequency_penalty: frequencyPenalty,
      stream: stream,
    };

    let retryCount = 0;
    let lastError: any = null;

    while (retryCount <= this.maxRetries) {
      try {
        if (retryCount > 0) {
          const backoffMs = Math.min(1000 * Math.pow(2, retryCount - 1), 10000);
          await new Promise(resolve => setTimeout(resolve, backoffMs));

          errorHandler.handleError(
            new TaskError(
              `Z.ai API error (retry ${retryCount}/${this.maxRetries}): ${lastError instanceof Error ? lastError.message : String(lastError)}`,
              ErrorCategory.LLM,
              ErrorSeverity.WARNING,
              { operation: 'zai-complete', additionalInfo: { retry: retryCount } },
              lastError instanceof Error ? lastError : undefined
            ),
            true
          );
        }

        if (stream && onStreamUpdate) {
          const streamResponse = await this.client.chat.completions.create({
            ...params,
            stream: true,
          });

          let fullResponse = '';
          let finishReason: string | null = null;

          for await (const chunk of streamResponse) {
            const content = chunk.choices[0]?.delta?.content || '';
            if (content) {
              fullResponse += content;
              onStreamUpdate(content);
            }
            if (chunk.choices[0]?.finish_reason) {
              finishReason = chunk.choices[0].finish_reason;
            }
          }

          // Strip GLM thinking tags if present (GLM-4.7 may include <think> blocks)
          const cleanedResponse = fullResponse.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();

          return {
            text: cleanedResponse,
            usage: null,
            model: this.model,
            finishReason: finishReason || undefined,
          };
        } else {
          const response = await this.client.chat.completions.create({
            ...params,
            stream: false,
          });

          let text = response.choices[0]?.message?.content || '';

          // Strip GLM thinking tags if present (GLM-4.7 may include <think> blocks)
          // This matches the codebase's handling in taskManager.ts
          text = text.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();

          const usage: LLMUsage | null = response.usage
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
      } catch (error) {
        lastError = error;

        const isRetryable =
          error instanceof Error &&
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

        errorHandler.handleError(
          new TaskError(
            `Z.ai API error: ${error instanceof Error ? error.message : String(error)}`,
            ErrorCategory.LLM,
            ErrorSeverity.ERROR,
            { operation: 'zai-complete' },
            error instanceof Error ? error : undefined
          )
        );

        throw new TaskError(
          `Z.ai API error: ${error instanceof Error ? error.message : String(error)}`,
          ErrorCategory.LLM,
          ErrorSeverity.ERROR,
          { operation: 'zai-complete' },
          error instanceof Error ? error : undefined
        );
      }
    }

    errorHandler.handleError(
      new TaskError(
        `Z.ai API error (after ${this.maxRetries} retries): ${lastError instanceof Error ? lastError.message : String(lastError)}`,
        ErrorCategory.LLM,
        ErrorSeverity.ERROR,
        { operation: 'zai-complete', additionalInfo: { maxRetriesExceeded: true } },
        lastError instanceof Error ? lastError : undefined
      )
    );

    throw new TaskError(
      `Z.ai API error (after ${this.maxRetries} retries): ${lastError instanceof Error ? lastError.message : String(lastError)}`,
      ErrorCategory.LLM,
      ErrorSeverity.ERROR,
      { operation: 'zai-complete', additionalInfo: { maxRetriesExceeded: true } },
      lastError instanceof Error ? lastError : undefined
    );
  }

  /**
   * Check if the provider is configured with an API key.
   * This is a synchronous check that does not verify connectivity.
   */
  isAvailable(): boolean {
    return !!this.apiKey;
  }

  /**
   * Test the connection to Z.ai API by making a minimal request.
   * Returns true if the connection is successful, false otherwise.
   * Useful for verifying API key validity and network connectivity.
   */
  async testConnection(): Promise<{ success: boolean; message: string; latencyMs?: number }> {
    const startTime = Date.now();

    try {
      const response = await this.client.chat.completions.create({
        model: this.model,
        messages: [{ role: 'user', content: 'Hi' }],
        max_tokens: 5,
        temperature: 0,
      });

      const latencyMs = Date.now() - startTime;
      const hasContent = !!response.choices[0]?.message?.content;

      return {
        success: hasContent,
        message: hasContent
          ? `Connection successful. Model: ${response.model}`
          : 'Connection established but no response content received',
        latencyMs,
      };
    } catch (error) {
      const latencyMs = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : String(error);

      errorHandler.handleError(
        new TaskError(
          `Z.ai connection test failed: ${errorMessage}`,
          ErrorCategory.LLM,
          ErrorSeverity.WARNING,
          { operation: 'zai-testConnection' },
          error instanceof Error ? error : undefined
        ),
        true
      );

      return {
        success: false,
        message: `Connection failed: ${errorMessage}`,
        latencyMs,
      };
    }
  }

  getProviderName(): string {
    return 'Z.ai';
  }

  getModelName(): string {
    return this.model;
  }
}
