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
 * - ZAI_ENABLE_THINKING: Set to "true" to enable thinking mode (default: disabled)
 *
 * Available Text Models:
 * - glm-4.7 (recommended) - 200K context, 128K output, temp default 1.0
 * - glm-4.7-flash - Faster variant
 * - glm-4.7-flashx - Extended flash variant
 * - glm-4.6 - Previous generation, temp default 1.0
 * - glm-4.5 - temp default 0.6, 96K max output
 * - glm-4.5-air - Lighter variant
 * - glm-4.5-x, glm-4.5-airx, glm-4.5-flash - Extended variants
 * - glm-4-32b-0414-128k - 32B parameter model, 16K max output
 *
 * Available Vision Models:
 * - glm-4.6v - 32K max output
 * - glm-4.6v-flash, glm-4.6v-flashx
 * - glm-4.5v - 16K max output
 * - autoglm-phone-multilingual - 4K max output, temp default 0.0
 *
 * Z.ai-Specific API Features:
 * - response_format: {"type": "json_object"} for structured JSON output
 * - thinking: {type: "enabled"|"disabled"} for chain-of-thought control
 * - do_sample: Boolean - when false, disables temperature/top_p for deterministic output
 * - tools: Function calling with max 128 functions, includes web_search
 * - tool_choice: "auto" for automatic function selection
 * - stop: String array (max 1 item) for generation terminators
 *
 * Finish Reasons:
 * - stop: Normal completion
 * - tool_calls: Model wants to call a function
 * - length: Hit max_tokens limit
 * - sensitive: Content moderation triggered
 * - network_error: Network issue (retriable)
 *
 * Note: presence_penalty and frequency_penalty are passed through but may not
 * be supported by Z.ai API (not documented). They may be silently ignored.
 */

/**
 * Extended parameters for Z.ai API that aren't in standard OpenAI SDK
 */
interface ZaiExtendedParams {
  response_format?: { type: 'text' | 'json_object' };
  thinking?: { type: 'enabled' | 'disabled'; clear_thinking?: boolean };
  do_sample?: boolean;
  stop?: string[];
}

export class ZaiClient implements LLMClient {
  private client: OpenAI;
  private model: string;
  private maxRetries: number = 3;
  private apiKey?: string;
  private baseURL: string;
  private enableThinking: boolean;

  constructor(model?: string, apiKey?: string, baseURL?: string) {
    this.apiKey = apiKey || process.env.ZAI_API_KEY;
    this.baseURL = baseURL || process.env.ZAI_BASE_URL || 'https://api.z.ai/api/coding/paas/v4';
    this.enableThinking = process.env.ZAI_ENABLE_THINKING === 'true';

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
      temperature = 1.0, // GLM-4.7 official default per Z.ai docs
      topP,
      presencePenalty,
      frequencyPenalty,
      stream,
      onStreamUpdate,
      systemPrompt,
      stopSequences
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
      temperature: temperature,
      top_p: topP,
      // Note: presence_penalty and frequency_penalty may not be supported by Z.ai
      // They are passed through but may be silently ignored
      presence_penalty: presencePenalty,
      frequency_penalty: frequencyPenalty,
      stream: stream,
    };

    // Add Z.ai-specific parameters
    const zaiParams: ZaiExtendedParams = {};

    // Use official response_format for JSON mode
    if (isJsonRequest) {
      zaiParams.response_format = { type: 'json_object' };
      // Use do_sample: false for deterministic JSON output (official Z.ai method)
      zaiParams.do_sample = false;
    }

    // Thinking mode: disabled by default for cleaner output, configurable via env
    if (!this.enableThinking) {
      zaiParams.thinking = { type: 'disabled' };
    }

    // Z.ai only supports 1 stop sequence (max items: 1)
    if (stopSequences && stopSequences.length > 0) {
      zaiParams.stop = [stopSequences[0]];
      if (stopSequences.length > 1) {
        errorHandler.handleError(
          new TaskError(
            `Z.ai only supports 1 stop sequence. Using first: "${stopSequences[0]}". Ignored: ${stopSequences.slice(1).join(', ')}`,
            ErrorCategory.LLM,
            ErrorSeverity.WARNING,
            { operation: 'zai-complete' }
          ),
          true
        );
      }
    }

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

          // Handle Z.ai-specific finish reasons
          this.handleFinishReason(finishReason);

          // Fallback: strip thinking tags if API param didn't prevent them
          const cleanedResponse = this.enableThinking
            ? fullResponse
            : fullResponse.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();

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
          const finishReason = response.choices[0]?.finish_reason;

          // Handle Z.ai-specific finish reasons
          this.handleFinishReason(finishReason);

          // Fallback: strip thinking tags if API param didn't prevent them
          if (!this.enableThinking) {
            text = text.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
          }

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
            finishReason: finishReason || undefined,
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
          errorMessage.includes('network_error');

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
   * Handle Z.ai-specific finish reasons
   * - sensitive: Content moderation triggered
   * - length: Output was truncated
   * - tool_calls: Model wants to call a function
   */
  private handleFinishReason(finishReason: string | null | undefined): void {
    if (!finishReason) return;

    switch (finishReason) {
      case 'sensitive':
        errorHandler.handleError(
          new TaskError(
            'Z.ai content moderation triggered. The response may be incomplete or filtered.',
            ErrorCategory.LLM,
            ErrorSeverity.WARNING,
            { operation: 'zai-complete', additionalInfo: { finishReason: 'sensitive' } }
          ),
          true
        );
        break;
      case 'length':
        errorHandler.handleError(
          new TaskError(
            'Z.ai response was truncated due to max_tokens limit.',
            ErrorCategory.LLM,
            ErrorSeverity.WARNING,
            { operation: 'zai-complete', additionalInfo: { finishReason: 'length' } }
          ),
          true
        );
        break;
      case 'tool_calls':
        // This is informational - the model wants to call a function
        // Logging at debug level since function calling isn't used in this codebase
        break;
    }
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
   * Uses thinking: disabled and do_sample: false for fast, deterministic response.
   */
  async testConnection(): Promise<{ success: boolean; message: string; latencyMs?: number }> {
    const startTime = Date.now();

    try {
      const response = await this.client.chat.completions.create({
        model: this.model,
        messages: [{ role: 'user', content: 'Hi' }],
        max_tokens: 5,
        thinking: { type: 'disabled' },
        do_sample: false,
      } as OpenAI.Chat.ChatCompletionCreateParams);

      const latencyMs = Date.now() - startTime;
      const hasContent = !!response.choices[0]?.message?.content;
      const finishReason = response.choices[0]?.finish_reason;

      // Check for content moderation
      if (finishReason === 'sensitive') {
        return {
          success: false,
          message: 'Connection successful but test message was filtered by content moderation',
          latencyMs,
        };
      }

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
