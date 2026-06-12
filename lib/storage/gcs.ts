import { Storage } from '@google-cloud/storage';

// Initialize Google Cloud Storage
const storage = new Storage({
  projectId: process.env.GOOGLE_CLOUD_PROJECT,
  keyFilename: process.env.GOOGLE_APPLICATION_CREDENTIALS,
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
