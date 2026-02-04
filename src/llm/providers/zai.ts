import { LLMClient, LLMCompletionOptions, LLMCompletionResult, LLMUsage } from '../types.js';
import OpenAI from 'openai';
import process from 'process';
import { ErrorHandler, ErrorCategory, ErrorSeverity, TaskError } from '../../core/errorHandler.js';

const errorHandler = ErrorHandler.getInstance();

/**
 * Z.ai GLM Coding Plan Client
 *
 * Uses OpenAI-compatible API format with Z.ai's GLM models.
 *
 * Documentation:
 * - Overview: https://docs.z.ai/devpack/overview
 * - Chat Completion API: https://docs.z.ai/api-reference/llm/chat-completion
 * - Function Calling: https://docs.z.ai/guides/capabilities/function-calling
 * - Structured Output: https://docs.z.ai/guides/capabilities/struct-output
 *
 * Environment Variables:
 * - ZAI_API_KEY: Required API key from Z.ai
 * - ZAI_MODEL: Model to use (default: glm-4.7)
 * - ZAI_BASE_URL: Custom base URL (default: https://api.z.ai/api/coding/paas/v4)
 *
 * Available Models:
 * - glm-4.7 (recommended, latest) - 200K context, 128K output, default temp 1.0
 * - glm-4.7-flash - Faster variant
 * - glm-4.6 - Previous generation
 * - glm-4.5 - Default temp 0.6
 * - glm-4.5-air (faster, lighter)
 *
 * Z.ai-Specific API Features:
 * - response_format: {"type": "json_object"} for structured JSON output
 * - thinking: {type: "enabled"|"disabled"} for chain-of-thought control
 * - tools: Function calling with max 128 functions
 * - tool_choice: "auto" for automatic function selection
 * - do_sample: Boolean to enable/disable sampling (disables temp/top_p when false)
 *
 * GLM-4.7 Capabilities:
 * - Function/tool calling (OpenAI format)
 * - Interleaved thinking mode
 * - Strong coding benchmarks (LiveCodeBench-v6: 84.9, SWE-bench: 73.8%)
 */

/**
 * Extended parameters for Z.ai API that aren't in standard OpenAI SDK
 */
interface ZaiExtendedParams {
  response_format?: { type: 'text' | 'json_object' };
  thinking?: { type: 'enabled' | 'disabled'; clear_thinking?: boolean };
  do_sample?: boolean;
}

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
      temperature = 1.0, // GLM-4.7 default is 1.0 per official docs
      topP,
      presencePenalty,
      frequencyPenalty,
      stream,
      onStreamUpdate,
      systemPrompt
    } = options;

    // Detect JSON requests to use official response_format parameter
    const isJsonRequest = systemPrompt?.includes('JSON') ||
                          systemPrompt?.includes('json') ||
                          prompt?.includes('JSON') ||
                          prompt?.includes('json');

    // Build base params
    const baseParams: OpenAI.Chat.ChatCompletionCreateParams = {
      model: this.model,
      messages: systemPrompt
        ? [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: prompt }
          ]
        : [{ role: 'user', content: prompt }],
      max_tokens: maxTokens,
      temperature: isJsonRequest ? Math.min(temperature, 0.3) : temperature, // Lower temp for JSON reliability
      top_p: topP,
      presence_penalty: presencePenalty,
      frequency_penalty: frequencyPenalty,
      stream: stream,
    };

    // Add Z.ai-specific parameters using type assertion
    const zaiParams: ZaiExtendedParams = {};

    // Use official response_format for JSON mode (per Z.ai docs)
    if (isJsonRequest) {
      zaiParams.response_format = { type: 'json_object' };
    }

    // Disable thinking mode to prevent <think> blocks in output
    // This uses the official API parameter instead of regex stripping
    zaiParams.thinking = { type: 'disabled' };

    // Merge params (Z.ai API accepts these additional fields)
    const params = { ...baseParams, ...zaiParams } as OpenAI.Chat.ChatCompletionCreateParams;

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

          // Fallback: strip thinking tags if API param didn't prevent them
          // (defensive coding in case thinking param is ignored)
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

          // Fallback: strip thinking tags if API param didn't prevent them
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

        // Handle Z.ai-specific error conditions
        const errorMessage = error instanceof Error ? error.message.toLowerCase() : String(error).toLowerCase();

        const isRetryable =
          errorMessage.includes('network') ||
          errorMessage.includes('timeout') ||
          errorMessage.includes('rate') ||
          errorMessage.includes('limit') ||
          errorMessage.includes('429') ||
          errorMessage.includes('500') ||
          errorMessage.includes('502') ||
          errorMessage.includes('503') ||
          errorMessage.includes('504') ||
          errorMessage.includes('network_error'); // Z.ai-specific finish_reason

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
   * Uses thinking: disabled to minimize response overhead.
   */
  async testConnection(): Promise<{ success: boolean; message: string; latencyMs?: number }> {
    const startTime = Date.now();

    try {
      // Use type assertion for Z.ai-specific params
      const response = await this.client.chat.completions.create({
        model: this.model,
        messages: [{ role: 'user', content: 'Hi' }],
        max_tokens: 5,
        temperature: 0,
        thinking: { type: 'disabled' },
      } as OpenAI.Chat.ChatCompletionCreateParams);

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
