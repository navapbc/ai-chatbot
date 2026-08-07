import { Storage } from '@google-cloud/storage';
import { Compute } from 'google-auth-library';

/** Write scope for replay uploads; Compute clients need it stated explicitly. */
const STORAGE_SCOPE = 'https://www.googleapis.com/auth/devstorage.read_write';

// Initialize Google Cloud Storage.
//
// The bucket grants storage.objectAdmin to the per-env runtime service account
// (cloud-run-<env>@…) in terraform/storage.tf, and that is the identity Cloud
// Run attaches to the container. But Cloud Run also sets
// GOOGLE_APPLICATION_CREDENTIALS to the mounted Vertex AI key file so the AI
// SDK can reach Vertex — and that variable overrides the attached identity for
// every Google client in the process. Storage was therefore authenticating as
// vertex-ai@, which has no bucket access, and every replay upload failed with
// "does not have storage.objects.create access" (silently: archival is
// best-effort). Replays stopped landing in the bucket on 2026-07-23.
//
// Pin Storage to the attached runtime service account with a Compute client,
// which reads the GCE/Cloud Run metadata server directly and so never consults
// GOOGLE_APPLICATION_CREDENTIALS. Vertex keeps using the key file, which is
// what it needs. Off Cloud Run (local dev, tests) there is no metadata server,
// so fall back to normal ADC.
const storage = new Storage({
  projectId: process.env.GOOGLE_CLOUD_PROJECT,
  ...(process.env.K_SERVICE
    ? { authClient: new Compute({ scopes: STORAGE_SCOPE }) }
    : {}),
});

const bucketName = process.env.GCS_BUCKET_NAME || 'nava-storage-dev';
const bucket = storage.bucket(bucketName);

export interface UploadResult {
  url: string;
  downloadUrl: string;
  pathname: string;
  size: number;
}

export async function uploadFile(
  filename: string,
  fileBuffer: ArrayBuffer,
  contentType: string,
): Promise<UploadResult> {
  try {
    // Generate a unique filename to avoid conflicts
    const timestamp = Date.now();
    const uniqueFilename = `chat-uploads/${timestamp}-${filename}`;

    const file = bucket.file(uniqueFilename);

    // Upload the file
    await file.save(Buffer.from(fileBuffer), {
      metadata: {
        contentType,
      },
      public: true, // Make file publicly accessible
    });

    // Get the public URL
    const publicUrl = `https://storage.googleapis.com/${bucketName}/${uniqueFilename}`;

    return {
      url: publicUrl,
      downloadUrl: publicUrl,
      pathname: uniqueFilename,
      size: fileBuffer.byteLength,
    };
  } catch (error) {
    console.error('Error uploading file to GCS:', error);
    throw new Error('Failed to upload file to Google Cloud Storage');
  }
}

export async function deleteFile(pathname: string): Promise<void> {
  try {
    await bucket.file(pathname).delete();
  } catch (error) {
    console.error('Error deleting file from GCS:', error);
    throw new Error('Failed to delete file from Google Cloud Storage');
  }
}

export async function getFileUrl(pathname: string): Promise<string> {
  return `https://storage.googleapis.com/${bucketName}/${pathname}`;
}

/**
 * Upload a kernel browser replay video to object storage under a stable,
 * chat-scoped path: `replays/<chatId>/<kernelSessionId>.mp4`. Deterministic
 * (no timestamp) so a re-run overwrites rather than duplicates. Returns the
 * stored URL to persist on the session mapping.
 *
 * Note: objects under `replays/` should be excluded from the bucket's default
 * deletion lifecycle rule (see terraform/storage.tf) so videos are retained.
 */
export async function uploadReplayVideo(
  chatId: string,
  kernelSessionId: string,
  videoBuffer: ArrayBuffer,
): Promise<string> {
  const pathname = `replays/${chatId}/${kernelSessionId}.mp4`;
  const file = bucket.file(pathname);

  await file.save(Buffer.from(videoBuffer), {
    metadata: { contentType: 'video/mp4' },
  });

  return `https://storage.googleapis.com/${bucketName}/${pathname}`;
}
